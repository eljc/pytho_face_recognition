from wsgiref.simple_server import make_server

def web_app(enviroment, response):
    status = '200 OK'
    headers = [('content-type', 'text/html; charset=utf-8')]
    response(status, headers)

    return [b'<strong>No Ar</strong>']

with make_server('', 5001, web_app) as server:
    print('Start Servico: porta: 5001..\nAcesso http://127.0.0.1:5001\nStop Ctrl + C')

    server.serve_forever()