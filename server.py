import socket
s = socket.socket()
port = 3018
host = ""
s.bind((host,port))
s.listen(5)
print("server listening")

while True:
	conn,addr = s.accept()
	print("connected address: ", addr)
	data = conn.recv(1024)
	message = "hello " + str(addr[0])
	print("received data: ",data)
	conn.send(message)
	print("message sent")
	conn.close()
