import socket
import json

class Server():
    def __init__(self, host = '127.0.0.1', port = 5055, listener = 5):
        self.host = host
        self.port = port
        self.lisetener = listener
        self.bufferSize = 12000
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # family:server to server; type:TCP
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind((host, self.port))
        # self.s.setblocking(False) # non blocking
        self.s.listen(self.lisetener) # at most how many sockets connect
        print('server start at: %s:%s' % (self.host, self.port))
        print('wait for connection...')

        self.client, addr = self.s.accept() # wait for client request
        print('connected by' + str(addr))

    def recvData(self): #while true??
        indata = self.client.recv(self.bufferSize).decode()
        indata = json.loads(indata)
        # print('Receive from Unity: ', indata)
        return indata

    def sendAction(self, action):
        # print(action)
        self.client.send(json.dumps(action, indent = 4).encode())
        # print('send action to Unity...', action)
    
    def close(self):
        self.client.close()