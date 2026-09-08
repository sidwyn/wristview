#!/usr/bin/env python3
"""slop_check.py: flag AI-writing tells in a markdown file.

Usage:  python tools/slop_check.py essay/wristview-essay-sid.md [--summary]

Two kinds of check:
  1. Phrase tells. Words and phrases that LLMs over-produce. Sources: Wikipedia
     "Signs of AI writing", Peter Yang's no-ai-slop list, Hardik Pandya's stop-slop,
     the 2026 blacklists (oliviacal, metric37). Each hit prints the line.
  2. Structure tells. Things a word list cannot catch: negative parallelism
     ("not X, it's Y"), rule-of-three lists, em-dash reveals, colon reveals,
     dramatic fragments, sentence-length monotony, punchy one-line paragraph
     closers, bold-label bullets, "-ing" analysis tails, vague attribution,
     section-restating closers, hedges, and knowledge of the reader ("you").
  3. Voice checks (optional, --voice). Compares the document against the author's
     observed habits from his Path to Staff essays: contraction rate, first-person
     rate, question-heading rate, "So"/"Now" pivots, mean sentence length.

Exit code is the number of HIGH-severity hits, so it can gate a build.
Nothing here is authoritative. Each hit is a place to look, not a verdict.
"""
import re, sys, statistics, argparse

# ---------- 1. phrase tells -----------------------------------------------
# (pattern, severity, note)
PHRASES = [
    # throat-clearing / openers
    (r"\bhere'?s the thing\b", "high", "throat-clearing opener"),
    (r"\blet me be clear\b", "high", "throat-clearing opener"),
    (r"\blet'?s (dive|delve) (in|into)\b", "high", "opener cliché"),
    (r"\bin today'?s (fast-paced|digital|ever-changing|rapidly)", "high", "opener cliché"),
    (r"\bpicture this\b", "med", "fake-experience opener"),
    (r"\bimagine (a world|if)\b", "med", "fake-experience opener"),
    # faux-insight setups
    (r"\bwhat (nobody|no one) tells you\b", "high", "faux-insight setup"),
    (r"\bthe part (everyone|most people) miss(es)?\b", "high", "faux-insight setup"),
    (r"\bthe (real|hard|uncomfortable) truth is\b", "high", "faux-insight setup"),
    (r"\bthe (best|worst|interesting|surprising|key) part[:?]", "med", "colon reveal setup"),
    # editorialising asides
    (r"\bit'?s (important|worth|crucial|essential) to (note|remember|consider|mention|understand)\b", "high", "editorialising aside"),
    (r"\bit (should|must) be noted\b", "high", "editorialising aside"),
    (r"\bno discussion .* would be complete\b", "high", "editorialising aside"),
    (r"\bneedless to say\b", "med", "editorialising aside"),
    (r"\bin summary\b|\bin conclusion\b|\bto sum up\b|\ball in all\b", "med", "section-restating closer"),
    (r"\bat the end of the day\b", "med", "cliché closer"),
    # importance puffery / inflated symbolism
    (r"\b(a |stands as a )?testament to\b", "high", "puffery"),
    (r"\b(pivotal|watershed|defining|seminal) (moment|role)\b", "high", "puffery"),
    (r"\bvital role\b|\bcrucial role\b|\bkey role\b", "med", "puffery"),
    (r"\blasting (impact|legacy)\b|\bprofound (impact|heritage)\b", "high", "puffery"),
    (r"\bgame[- ]?chang(er|ing)\b|\bparadigm shift\b|\brevolutioni[sz]e", "high", "puffery"),
    (r"\btransformative\b|\bgroundbreaking\b|\bcutting[- ]edge\b|\bstate[- ]of[- ]the[- ]art\b", "med", "puffery"),
    (r"\bunlock(s|ed|ing)? (the )?(power|potential|value)\b", "high", "puffery"),
    (r"\b(rich|vibrant) (tapestry|heritage|ecosystem)\b|\btapestry\b", "high", "flowery metaphor"),
    (r"\bbeacon\b|\bsymphony\b|\brealm\b|\blandscape of\b", "med", "flowery metaphor"),
    (r"\bjourney\b", "low", "flowery metaphor (fine if literal)"),
    (r"\bnavigat(e|ing) the (complex|complexities|landscape|world)\b", "high", "flowery metaphor"),
    (r"\bstunning\b|\bbreathtaking\b|\bmust[- ]visit\b|\bnestled\b", "med", "promotional"),
    # buzzwords
    (r"\bdelve(s|d)?\b", "high", "AI vocabulary"),
    (r"\bleverag(e|es|ing)\b", "high", "AI vocabulary (use 'use')"),
    (r"\butili[sz](e|es|ing)\b", "med", "AI vocabulary (use 'use')"),
    (r"\bseamless(ly)?\b|\brobust\b|\bstreamlin(e|ed|ing)\b", "med", "AI vocabulary"),
    (r"\bempower(s|ed|ing)?\b|\bsynerg(y|ies)\b|\bholistic\b", "high", "AI vocabulary"),
    (r"\bmyriad\b|\bplethora\b|\bmultifaceted\b|\bcomprehensive\b", "med", "AI vocabulary"),
    (r"\bmeticulous(ly)?\b|\bintricate\b|\bnuanced\b", "med", "AI vocabulary"),
    (r"\bfoster(s|ed|ing)?\b|\bunderscore(s|d)?\b|\bshowcase(s|d)?\b", "med", "AI vocabulary"),
    (r"\bever[- ]evolving\b|\bfast[- ]paced\b|\brapidly (changing|evolving)\b", "high", "AI vocabulary"),
    (r"\bat its core\b|\bat the heart of\b|\bin the realm of\b", "med", "AI vocabulary"),
    (r"\bdemystif(y|ies|ied)\b|\bdeep dive\b", "med", "AI vocabulary"),
    (r"\bgranular\b|\bactionable\b|\bimpactful\b", "med", "AI vocabulary"),
    (r"\ba (wide|broad|diverse) (range|array) of\b|\baligns? with\b", "med", "AI vocabulary"),
    # transitions
    (r"^\s*(furthermore|moreover|additionally|consequently|nevertheless|notably|importantly|ultimately|crucially|interestingly),", "med", "conjunctive opener"),
    (r"^\s*(that said|with that said|that being said|in essence|in other words),", "low", "conjunctive opener"),
    # vague attribution
    (r"\b(experts|studies|research|industry reports|observers|critics|many) (agree|show|suggest|say|argue|cite|believe)\b", "high", "vague attribution"),
    (r"\bit is (widely|generally|commonly) (known|accepted|believed)\b", "high", "vague attribution"),
    # hedging
    (r"\baims? to\b", "low", "hedge"),
    (r"\bcan (potentially|possibly) \b|\bmay (potentially|possibly)\b", "med", "double hedge"),
    (r"\bplays? a (role|part) in\b", "med", "vague verb"),
    (r"\bserves? as\b|\bfunctions? as\b", "low", "vague verb"),
    # collaborative leftovers
    (r"\bI hope this (helps|finds you)\b|\bfeel free to reach out\b|\blet me know if you (have|need) any\b", "high", "chatbot leftover"),
    (r"\bas an ai(,| language| model| assistant)|\bas of my (last|knowledge)\b", "high", "chatbot leftover"),
    # fake-profound endings
    (r"\bthe future (is|isn'?t) (here|now|coming)\b|\bonly time will tell\b", "high", "fake-profound ending"),
    (r"\bthe (real )?question (isn'?t|is not) whether\b", "high", "fake-profound framing"),
    (r"\bthat'?s (it|the whole thing|the point|all)\.", "med", "dramatic fragment"),
    (r"\bfull stop\.|\bperiod\.\s*$", "med", "dramatic fragment"),
    (r"\bnot because .* but because\b", "med", "contrast reframe"),
    # false range
    (r"\b(ranging|ranges|range) from .{3,40} to\b", "low", "false range (check it means something)"),
    (r"\b(everything|anything) from .{3,40} to\b", "med", "false range"),
    # LLM sentence tics
    (r"\bin a world where\b|\bin an era of\b|\bin the age of\b", "high", "LLM tic"),
    (r"\bwhether you'?re a .* or a\b", "high", "LLM tic"),
    (r"\bthis (isn'?t|is not) (just|about)\b", "med", "negative parallelism"),
    (r"\bmore than (just )?(a|an)\b", "med", "negative parallelism"),
]

# ---------- 2. structure tells ---------------------------------------------
NEG_PAR = re.compile(r"\b(not|isn'?t|aren'?t|wasn'?t|weren'?t|don'?t|doesn'?t)( just| only| simply| merely)?\b[^.;]{2,60}?[,;.]\s*(it'?s|it is|but|they'?re|but rather|rather)\b", re.I)
NOT_ONLY = re.compile(r"\bnot only\b.{3,80}?\bbut( also)?\b", re.I)
TRIPLET_ADJ = re.compile(r"\b(\w+), (\w+),? and (\w+)\b")
EM_DASH = re.compile(r"—")
SPACED_DASH = re.compile(r"\s[-–]\s")
COLON_REVEAL = re.compile(r"^[^:]{8,60}:\s+[A-Za-z][^:]{3,80}\.$")
BOLD_BULLET = re.compile(r"^\s*[-*]\s+\*\*[^*]+\*\*:")
ING_TAIL = re.compile(r",\s+(ensuring|highlighting|reflecting|showcasing|underscoring|emphasizing|emphasising|demonstrating|illustrating|signaling|signalling|marking|solidifying|cementing)\b[^.]*\.$", re.I)
QUOTABLE = re.compile(r"^[A-Z][^.!?]{8,70}\.$")  # short declarative one-liner
HEADING = re.compile(r"^#{1,6}\s+(.*)")
TITLE_CASE_HEAD = re.compile(r"^(?:[A-Z][a-z]+\s+){2,}[A-Z][a-z]+$")

def sentences(text):
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", text)
    out = []
    for p in parts:
        p = p.strip()
        if len(p) <= 1: continue
        if re.fullmatch(r"\(?[^()]{0,20}\)?\.?", p) and p.startswith("("): continue  # "(code pointer)"
        if re.fullmatch(r"\**\d+\.\**", p): continue  # list marker "**1."
        out.append(p)
    return out

def words(s):
    return re.findall(r"[A-Za-z']+", s)

def strip_md(line):
    line = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", line)
    line = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", line)
    line = re.sub(r"`[^`]*`", "", line)
    line = re.sub(r"<[^>]+>", "", line)
    return line

def paragraphs(lines):
    """yield (start_line_no, text) for prose paragraphs; skip code, tables, headings, figures"""
    buf, start, in_code = [], None, False
    for i, raw in enumerate(lines, 1):
        if raw.strip().startswith("```"):
            in_code = not in_code; continue
        if in_code: continue
        s = raw.strip()
        if not s or s.startswith(("#", "|", "![", "<", "_", "---")):
            if buf:
                yield start, " ".join(buf); buf = []
            continue
        if not buf: start = i
        buf.append(strip_md(s))
    if buf: yield start, " ".join(buf)

def check(path, voice=False, summary=False):
    lines = open(path, encoding="utf-8").read().split("\n")
    hits = []  # (severity, line, note, excerpt)

    # phrase tells, per line
    in_code = False
    for i, raw in enumerate(lines, 1):
        if raw.strip().startswith("```"): in_code = not in_code; continue
        if in_code or raw.strip().startswith(("|", "![", "<")): continue
        line = strip_md(raw)
        for pat, sev, note in PHRASES:
            for m in re.finditer(pat, line, re.I | re.M):
                hits.append((sev, i, note, snippet(line, m.start())))
        if BOLD_BULLET.match(raw):
            hits.append(("low", i, "bold-label bullet (fine for timelines, not for arguments)", raw.strip()[:70]))
        for m in EM_DASH.finditer(line):
            hits.append(("high", i, "em dash (your rule: none in the essay)", snippet(line, m.start())))
        for m in SPACED_DASH.finditer(line):
            hits.append(("low", i, "spaced dash (you use these; check it is not a reveal)", snippet(line, m.start())))
        h = HEADING.match(raw)
        if h and TITLE_CASE_HEAD.match(h.group(1).strip()):
            hits.append(("low", i, "Title Case heading", h.group(1)[:70]))

    # structure tells, per paragraph
    all_sents = []
    paras = list(paragraphs(lines))
    for n, (ln, text) in enumerate(paras):
        sents = sentences(text)
        all_sents += sents
        for s in sents:
            if NEG_PAR.search(s):
                hits.append(("high", ln, "negative parallelism (not X, it's Y)", s[:90]))
            if NOT_ONLY.search(s):
                hits.append(("med", ln, "not only / but also", s[:90]))
            if COLON_REVEAL.match(s):
                hits.append(("low", ln, "colon reveal (you use these; check the payoff is real)", s[:90]))
            if ING_TAIL.search(s):
                hits.append(("high", ln, "'-ing' analysis tail", s[:90]))
            m = TRIPLET_ADJ.search(s)
            if m and all(len(w) > 3 for w in m.groups()) and not re.search(r"\d", m.group(0)):
                hits.append(("low", ln, "rule of three (check it is a real list)", m.group(0)))
            wc = len(words(s))
            if wc <= 4 and n_words_para(text) > 30 and s.rstrip(".!?").lower() not in ("i had no idea", "no", "yes"):
                hits.append(("low", ln, "dramatic fragment", s))
        # three consecutive same-length sentences
        lens = [len(words(s)) for s in sents]
        for k in range(len(lens) - 2):
            trio = lens[k:k+3]
            if max(trio) - min(trio) <= 2 and min(trio) >= 8:
                hits.append(("med", ln, f"three sentences of near-equal length {trio}", sents[k][:60]))
                break
        # punchy one-liner paragraph closer after a long paragraph
        if len(sents) >= 3 and len(words(sents[-1])) <= 6 and QUOTABLE.match(sents[-1]):
            hits.append(("med", ln, "punchy one-line closer (pull-quote shape)", sents[-1]))
        # single-sentence paragraph that reads like a pull quote
        if len(sents) == 1 and QUOTABLE.match(sents[0]) and 6 <= len(words(sents[0])) <= 14 and not sents[0].endswith("?"):
            hits.append(("low", ln, "one-sentence paragraph (pull-quote shape)", sents[0]))

    # document-level rhythm
    lens = [len(words(s)) for s in all_sents if words(s)]
    stats = {}
    if len(lens) > 20:
        mean = statistics.mean(lens); sd = statistics.pstdev(lens)
        stats.update(sentences=len(lens), mean_len=round(mean, 1), sd_len=round(sd, 1), cv=round(sd / mean, 2))
        if sd / mean < 0.45:
            hits.append(("med", 0, f"low burstiness: sentence length CV {sd/mean:.2f} (human prose is usually >0.5)", ""))
    text_all = " ".join(all_sents)
    W = len(words(text_all)) or 1
    contractions = len(re.findall(r"\b\w+'(t|s|re|ve|ll|d|m)\b", text_all))
    first_person = len(re.findall(r"\b(I|I'm|I'd|I've|me|my)\b", text_all))
    second_person = len(re.findall(r"\b(you|your|you're)\b", text_all, re.I))
    numbers = len(re.findall(r"\b\d[\d.,]*\s*(mm|cm|m|%|ms|Hz|fps|hours?|minutes?|seconds?|steps?|episodes?|demos?|frames?|dollars?|\$)|\$\d", text_all))
    stats.update(words=W,
                 contractions_per_1k=round(1000 * contractions / W, 1),
                 first_person_per_1k=round(1000 * first_person / W, 1),
                 second_person_per_1k=round(1000 * second_person / W, 1),
                 numbers_with_units_per_1k=round(1000 * numbers / W, 1))
    heads = [HEADING.match(l).group(1) for l in lines if HEADING.match(l)]
    qheads = sum(1 for h in heads if h.strip().endswith("?"))
    pivots = len(re.findall(r"(^|[.!?]\s+)(So|Now|But|And|Turns out)[, ]", text_all))
    stats.update(headings=len(heads), question_headings=qheads, so_now_pivots_per_1k=round(1000 * pivots / W, 1))

    if voice:
        # Sidwyn's observed baseline (Path to Staff essays, Sept 2026): see essay/VOICE.md
        # measured on "Hardware Behind AI" and "What's Inside an LLM": contractions 19-20/1k,
        # first person 3-7/1k, second person 7-8/1k, mean sentence 15.5-16.7 words, CV 0.61-0.63,
        # So/Now/But pivots 2.8-3.5/1k. Ranges below are those with slack for a first-person essay.
        base = dict(contractions_per_1k=(12, 30), first_person_per_1k=(2, 40), second_person_per_1k=(3, 15),
                    so_now_pivots_per_1k=(1.5, 8), mean_len=(12, 19), cv=(0.5, 0.9))
        for k, (lo, hi) in base.items():
            v = stats.get(k)
            if v is None: continue
            if v < lo: hits.append(("med", 0, f"voice: {k}={v}, below your usual range {lo}-{hi}", ""))
            if v > hi: hits.append(("low", 0, f"voice: {k}={v}, above your usual range {lo}-{hi}", ""))

    order = {"high": 0, "med": 1, "low": 2}
    hits.sort(key=lambda h: (order[h[0]], h[1]))
    counts = {s: sum(1 for h in hits if h[0] == s) for s in order}
    print(f"== {path}")
    print("   " + "  ".join(f"{k}={v}" for k, v in stats.items()))
    print(f"   hits: high={counts['high']} med={counts['med']} low={counts['low']}")
    if not summary:
        for sev, ln, note, ex in hits:
            print(f"  [{sev:4}] L{ln:<4} {note:45} | {ex}")
    return counts["high"]

def n_words_para(t): return len(words(t))

def snippet(line, pos, w=40):
    a = max(0, pos - w); b = min(len(line), pos + w)
    return ("…" if a else "") + line[a:b].strip() + ("…" if b < len(line) else "")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--voice", action="store_true", help="compare against Sidwyn's baseline ranges")
    ap.add_argument("--summary", action="store_true", help="counts only")
    a = ap.parse_args()
    total = 0
    for f in a.files:
        total += check(f, voice=a.voice, summary=a.summary)
    sys.exit(min(total, 255))
