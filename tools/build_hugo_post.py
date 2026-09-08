#!/usr/bin/env python3
"""Build the Hugo page bundle for sidwyn.com from the essay markdown.

    python3 tools/build_hugo_post.py
    python3 tools/build_hugo_post.py --src essay/wristview-essay-sid.md --out essay/hugo-post/ego2wrist

What it does:
  1. Reads the essay. Takes the H1 as the title. Writes PaperMod front matter.
  2. Copies every asset the essay references into the bundle, compressed:
       png/jpg/jpeg -> jpg, longest side 1800 px, quality 85
       gif          -> mp4 (h264, 15 fps)
       mp4          -> mp4 re-encoded (crf 26) if the source is over 1.5 MB
  3. Rewrites the asset paths in the markdown to the bundle-relative names.
  4. Turns the hero GIF into an autoplaying <video>.

Run it after every edit of the essay. The bundle is generated; never edit
index.md by hand. Needs Pillow and ffmpeg on PATH.
"""
import argparse, os, re, shutil, subprocess, sys
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Appended to the post. Plays each clip when it scrolls into view and pauses it
# on the way out. A reader who pauses a clip by hand keeps it paused; scrolling
# past does not restart it. Readers who ask for reduced motion get no autoplay.
PLAY_IN_VIEW = """

<script>
(function () {
  var vids = document.querySelectorAll(".post-content video");
  if (!vids.length || !("IntersectionObserver" in window)) return;
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  function inView(v) {
    var r = v.getBoundingClientRect();
    return r.top < window.innerHeight && r.bottom > 0;
  }

  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      var v = e.target;
      if (!e.isIntersecting) { v.pause(); return; }
      if (v.dataset.held || v.ended) return;
      var p = v.play();
      if (p) p.catch(function () {});
    });
  }, { threshold: 0.25 });

  vids.forEach(function (v) {
    v.addEventListener("pause", function () {
      // Only a pause the reader asked for counts; ours happens off screen.
      if (!v.ended && inView(v)) v.dataset.held = "1";
    });
    v.addEventListener("play", function () { delete v.dataset.held; });
    io.observe(v);
  });
})();
</script>
"""

# The list-page and social-card image. Always copied into the bundle.
COVER_ASSET = "fig_hero_ego_vs_wrist.png"

# Used when the essay has no "Subtitle:" line after the H1.
DEFAULT_DESCRIPTION = (
    "I built a pipeline that fakes a robot's wrist camera from egocentric "
    "video. Then I tested whether the fakes were any good. My process, and "
    "lessons learned."
)

def sh(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"command failed: {' '.join(cmd)}\n{r.stderr}")

def convert(src, dst_dir):
    """Return the bundle file name for `src` after writing it into dst_dir."""
    name = os.path.basename(src); stem, ext = os.path.splitext(name); ext = ext.lower()
    if ext in (".png", ".jpg", ".jpeg"):
        out = f"{stem}.jpg"
        im = Image.open(src).convert("RGB")
        w, h = im.size; s = 1800 / max(w, h)
        if s < 1: im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
        im.save(os.path.join(dst_dir, out), quality=85, optimize=True)
    elif ext == ".gif":
        out = f"{stem}.mp4"
        sh("ffmpeg", "-y", "-v", "error", "-i", src, "-movflags", "+faststart", "-pix_fmt", "yuv420p",
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-r", "15", "-c:v", "libx264", "-crf", "26",
           os.path.join(dst_dir, out))
    elif ext in (".mp4", ".mov"):
        out = f"{stem}.mp4"
        if os.path.getsize(src) > 1_500_000 or ext == ".mov":
            sh("ffmpeg", "-y", "-v", "error", "-i", src, "-c:v", "libx264", "-crf", "26", "-preset", "slow",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", os.path.join(dst_dir, out))
        else:
            shutil.copy(src, os.path.join(dst_dir, out))
    else:
        out = name; shutil.copy(src, os.path.join(dst_dir, out))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "essay", "wristview-essay-sid.md"))
    ap.add_argument("--out", default=os.path.join(ROOT, "essay", "hugo-post", "ego2wrist"))
    ap.add_argument("--date", default="2026-09-09")
    a = ap.parse_args()

    md = open(a.src, encoding="utf-8").read()
    m = re.match(r"# (.*)\n", md)
    if not m: sys.exit("essay must start with an H1 line")
    title = m.group(1).strip().replace('"', '\\"'); body = md[m.end():].lstrip("\n")

    # An optional "Description:" (or "Subtitle:") line after the H1 sets the
    # front matter description. PaperMod prints it under the title as
    # .post-description. The text goes on the same line as the label, or in a
    # blockquote below it.
    description = DEFAULT_DESCRIPTION
    s = re.match(r"(?:Description|Subtitle):[ \t]*(.*)\n", body)
    if s:
        text, rest = s.group(1).strip(), body[s.end():]
        if not text:
            q = re.match(r"\n*((?:>[^\n]*\n)+)", rest)
            if q:
                text = " ".join(l.lstrip(">").strip()
                                for l in q.group(1).strip().split("\n")).strip()
                rest = rest[q.end():]
        if text: description = text.replace('"', '\\"')
        body = rest.lstrip("\n")

    leftovers = [l[:80] for l in body.split("\n") if re.search(r"CLAUDE|TODO|TOADD|TO ADD|<u>", l)]
    if leftovers:
        print("WARNING: placeholders still in the essay; they will be published as-is:")
        for l in leftovers: print("   ", l)

    if os.path.isdir(a.out):
        try:
            shutil.rmtree(a.out)
        except PermissionError:
            # Cowork's sandbox cannot delete files. Overwrite in place instead
            # and list anything left over that the essay no longer references.
            print("NOTE: cannot delete the old bundle here; overwriting in place")
    os.makedirs(a.out, exist_ok=True)
    stale = set(os.listdir(a.out))
    asset_dir = os.path.join(os.path.dirname(a.src), "assets")
    # The cover is named in the front matter, not the body, so carry it along
    # even when the essay does not show it inline.
    names = sorted(set(re.findall(r"assets/([A-Za-z0-9_.-]+)", body)) | {COVER_ASSET})
    outputs = {}
    for n in names:
        src = os.path.join(asset_dir, n)
        if not os.path.exists(src): sys.exit(f"missing asset: {src}")
        out = convert(src, a.out)
        outputs[n] = out
        stale.discard(out)
        body = body.replace("assets/" + n, out)
        print(f"  {n} -> {out}  {os.path.getsize(os.path.join(a.out, out)) // 1024} KB")

    # A converted gif becomes a video like any other.
    body = re.sub(r"!\[(.*?)\]\(([A-Za-z0-9_.-]+)\.mp4\)",
                  r'<video src="\2.mp4" controls muted playsinline width="100%" aria-label="\1"></video>', body)

    # Every video plays once when it scrolls into view, so drop "loop" and the
    # bare "autoplay" attribute; PLAY_IN_VIEW starts them instead. "controls"
    # stays: the clips run past 5s, so a reader needs a way to pause them.
    def video_tag(m):
        tag = re.sub(r"\s+(?:loop|autoplay)\b", "", m.group(0))
        if "preload=" not in tag:
            tag = tag.replace("<video", '<video preload="metadata"', 1)
        return tag
    body = re.sub(r"<video\b[^>]*>", video_tag, body)

    if "—" in body: print("WARNING: em dash found in the essay body")

    fm = (
        "---\n"
        f'title: "{title}"\n'
        f"date: {a.date}\n"
        "draft: false\n"
        'tags: ["robotics", "egocentric", "gaussian-splatting", "imitation-learning"]\n'
        f'description: "{description}"\n'
        "cover:\n"
        f'  image: "{outputs[COVER_ASSET]}"\n'
        '  alt: "Left: egocentric camera. Right: generated wrist video."\n'
        "  relative: true\n"
        "ShowToc: true\n"
        "TocOpen: false\n"
        "---\n\n"
    )
    open(os.path.join(a.out, "index.md"), "w", encoding="utf-8").write(fm + body + PLAY_IN_VIEW)
    stale.discard("index.md")
    for f in sorted(stale):
        try: os.remove(os.path.join(a.out, f))
        except PermissionError: print(f"WARNING: stale file the essay no longer uses, delete by hand: {f}")
    total = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out))
    print(f"bundle: {a.out}  {len(names)} media files  {total / 1e6:.1f} MB")

if __name__ == "__main__":
    main()
