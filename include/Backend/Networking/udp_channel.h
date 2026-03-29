// Copyright 2026 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
// associated documentation files (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#ifndef _GNET_UDP_CHANNEL
#define _GNET_UDP_CHANNEL

#include "../Database/GString.h"
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <pthread.h>

namespace GNet {

struct UDPPeer {
    sockaddr_storage addr;
    socklen_t addrLen;  // 0 = tombstoned (empty slot)
};

class UDPChannel {
public:
    UDPChannel();
    ~UDPChannel();

    bool Open(const shmea::GString& port);
    void Close();

    void Send(int peerIndex, const void* data, size_t len);
    void Broadcast(const void* data, size_t len);

    // Non-blocking receive. Returns bytes read, 0 if none, -1 on error.
    int Receive(void* buffer, size_t maxLen, sockaddr_storage& outAddr);

    // Returns stable peer index. Reuses tombstoned slots.
    int AddPeer(const sockaddr_storage& addr, socklen_t addrLen);
    void RemovePeer(int peerIndex);

    bool IsOpen() const { return m_socket >= 0; }

private:
    int m_socket;
    std::vector<UDPPeer> m_peers;
    pthread_mutex_t* m_peerMutex;
};

} // namespace GNet

#endif // _GNET_UDP_CHANNEL
