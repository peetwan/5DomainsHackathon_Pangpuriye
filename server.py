# Tiny static server for the index.html site (สำหรับ Railway / รันในเครื่อง)
# ใช้ stdlib ล้วน ไม่ต้องลงไลบรารีอะไรเพิ่ม
import http.server, socketserver, os

PORT = int(os.environ.get("PORT", "8000"))   # Railway จะตั้ง $PORT ให้เอง
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

socketserver.TCPServer.allow_reuse_address = True
print(f"serving 5Domains site on 0.0.0.0:{PORT}  -> open /index.html")
with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
    httpd.serve_forever()
