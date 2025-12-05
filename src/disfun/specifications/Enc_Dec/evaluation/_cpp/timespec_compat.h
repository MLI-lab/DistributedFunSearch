// Workaround for conda gcc 15.2 + glibc timespec_get incompatibility
#ifndef _TIMESPEC_COMPAT_H
#define _TIMESPEC_COMPAT_H

extern "C" {
    int timespec_get(struct timespec *ts, int base);
}

#endif
