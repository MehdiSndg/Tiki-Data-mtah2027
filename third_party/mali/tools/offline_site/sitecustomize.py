"""Opt-in outbound-network guard, loaded by Python before training imports.

Enabled with MTAH_OFFLINE=1 and this directory on PYTHONPATH. Local IPC is allowed;
outbound connections and remote DNS fail immediately instead of waiting for the internet.
"""

import os

if os.environ.get("MTAH_OFFLINE") == "1":
    import ipaddress
    import socket

    def _local(host):
        if isinstance(host, bytes):
            host = host.decode("ascii")
        if host == "localhost":
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    _connect = socket.socket.connect
    _connect_ex = socket.socket.connect_ex
    _getaddrinfo = socket.getaddrinfo

    def _check(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6) and not _local(address[0]):
            raise OSError("MTAH offline mode: outbound network is disabled")

    def connect(sock, address):
        _check(sock, address)
        return _connect(sock, address)

    def connect_ex(sock, address):
        _check(sock, address)
        return _connect_ex(sock, address)

    def getaddrinfo(host, *args, **kwargs):
        if host is not None and not _local(host):
            raise OSError("MTAH offline mode: remote DNS is disabled")
        return _getaddrinfo(host, *args, **kwargs)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.getaddrinfo = getaddrinfo
