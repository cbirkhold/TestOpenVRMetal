# TestOpenVR-Metal
Rendering to a Vive on macOS using OpenVR with either IOSurface backed Metal textures or Metal shared textures.

## IOSurface
OpenVR/SteamVR (as of 2018-08-24) still shares eye textures with its compositor process using shared memory (rather than mach ports or XPC) thus the kIOSurfaceIsGlobal property, although deprecated, must be set or submission will fail silently and the display will stay black.

## Metal shared textures
Metal from macOS 10.14 supports shared textures directly, including array textures, which can be submitted to OpenVR from 1.0.16.
