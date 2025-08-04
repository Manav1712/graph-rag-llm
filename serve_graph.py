#!/usr/bin/env python3
import http.server
import socketserver
import webbrowser
import os
import sys

def serve_graph():
    """Serve the knowledge graph visualization on localhost"""
    
    # Set the port
    PORT = 8000
    
    # Change to the directory containing the HTML file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create HTTP server
    Handler = http.server.SimpleHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"🌐 Knowledge Graph server running on http://localhost:{PORT}")
            print(f"📁 Serving files from: {os.getcwd()}")
            print(f"📊 Open http://localhost:{PORT}/knowledge_graph.html in your browser")
            print("Press Ctrl+C to stop the server")
            
            # Open browser automatically
            webbrowser.open(f'http://localhost:{PORT}/knowledge_graph.html')
            
            # Start serving
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {PORT} is already in use. Try a different port:")
            print(f"   python serve_graph.py --port 8001")
        else:
            print(f"❌ Error starting server: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Serve the knowledge graph visualization')
    parser.add_argument('--port', type=int, default=8000, help='Port to serve on (default: 8000)')
    
    args = parser.parse_args()
    
    # Check if required files exist
    if not os.path.exists('knowledge_graph.html'):
        print("❌ knowledge_graph.html not found in current directory")
        sys.exit(1)
    
    if not os.path.exists('graph_data.json'):
        print("❌ graph_data.json not found in current directory")
        sys.exit(1)
    
    print("✅ Required files found")
    print("🚀 Starting knowledge graph server...")
    
    serve_graph() 