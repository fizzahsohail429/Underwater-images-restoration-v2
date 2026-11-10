import re
import urllib.request

html = urllib.request.urlopen("https://li-chongyi.github.io/proj_benchmark.html", timeout=30).read().decode("utf-8", "replace")
for href, text in re.findall(r'href="([^"]+)"[^>]*>([^<]*)', html):
    low = (href + text).lower()
    if any(key in low for key in ("drive", "baidu", "github", "raw", "ref", "chall", "water")):
        print(text.strip(), "->", href)
