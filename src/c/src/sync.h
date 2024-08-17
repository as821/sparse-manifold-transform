#pragma once

#include "util.h"
#include <pthread.h>

struct SyncArgs {
    int loader_ptr;                     // last index processed by loader thread
    int consumer_ptr;                   // index currently being processed by main thread
    int worker_idx;                     // next index for a loader thread to consume
    pthread_mutex_t mutex;              // this actually only protects loader_ptr, consumer_ptr
    pthread_cond_t loader_wait;         // loader waits on this condvar until consumer ptr has progressed sufficiently
    pthread_cond_t consumer_wait;       // consumer waits on this condvar until loader has progressed
};


void producer_wait(struct SyncArgs* args, int64_t max_preload);
void producer_signal(struct SyncArgs* args, int64_t idx_completed);
void consumer_wait(struct SyncArgs* args);
void consumer_signal(struct SyncArgs* args);
int64_t loader_get_next_index(struct SyncArgs* args);

void sync_args_init(struct SyncArgs* args);


