#pragma once

#include "hvax/engine.hpp"

namespace httplib {
class Server;
}

namespace hvax {

// Register every route on an existing server. Tests bind the server to an
// ephemeral port and drive it with a client; hvaxd calls run_server.
void register_routes(Engine& engine, httplib::Server& svr);
void run_server(Engine& engine);

}  // namespace hvax
