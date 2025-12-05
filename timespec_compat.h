// Workaround for conda gcc 15.2 + glibc timespec_get incompatibility
#ifndef _TIMESPEC_COMPAT_H
#define _TIMESPEC_COMPAT_H

#include <time.h>

#ifndef timespec_get
#define timespec_get(ts, base) (-1)
#endif

#endif
