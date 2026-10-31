#!/bin/sh
# Diagnose the Irrawaddy Cloudflare block. Run on the VM.
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
echo "=== curl build ==="
curl --version | head -2
echo
echo "=== public IP ==="
curl -s --max-time 15 https://api.ipify.org; echo
echo
echo "=== targets (full browser headers) ==="
for u in "https://burma.irrawaddy.com/robots.txt" \
         "https://burma.irrawaddy.com/" \
         "https://burma.irrawaddy.com/sitemap_index.xml" \
         "https://burma.irrawaddy.com/article/2026/09/19/416361.html"; do
  out=$(curl -s -o /tmp/irrdiag.out -w '%{http_code} %{time_total}s' -L --max-time 25 \
    -A "$UA" \
    -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' \
    -H 'Accept-Language: en-US,en;q=0.9' \
    -H 'Upgrade-Insecure-Requests: 1' \
    -H 'Sec-Fetch-Dest: document' -H 'Sec-Fetch-Mode: navigate' -H 'Sec-Fetch-Site: none' \
    --compressed "$u")
  ch=$(grep -c 'Just a moment' /tmp/irrdiag.out 2>/dev/null || echo 0)
  echo "  $out challenge=$ch  $(echo $u | cut -c30-70)"
  sleep 3
done
echo
echo "=== control: other outlets reachable from this VM? ==="
for u in "https://myanmar-now.org/mm/" "https://burmese.dvb.no/"; do
  echo "  $(curl -s -o /dev/null -w '%{http_code}' -L --max-time 20 -A "$UA" "$u")  $u"
  sleep 2
done
