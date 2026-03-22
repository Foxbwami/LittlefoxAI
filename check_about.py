from pathlib import Path
p = Path('frontend/about.html')
s = p.read_text(encoding='utf-8', errors='replace')
print('utf8 contains:', '—' in s, '→' in s)
s2 = p.read_text(encoding='cp1252')
print('cp1252 contains:', '—' in s2, '→' in s2)
