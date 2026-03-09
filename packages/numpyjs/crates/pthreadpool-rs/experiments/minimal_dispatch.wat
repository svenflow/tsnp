;; Minimal futex dispatch test - hand-written WASM to find theoretical minimum overhead
;;
;; Memory layout (all offsets are i32 = 4 bytes):
;;   0: generation (i32, atomic) - bumped by main to signal workers
;;   4: active (i32, atomic) - decremented by threads when done
;;   8: n_threads (i32)
;;   12+: per-thread slots (not used in this minimal test)
;;
;; Protocol:
;;   Main: increment generation (release), notify all, spin on active==0
;;   Worker: spin/wait on generation change (acquire), decrement active (acqrel)

(module
  ;; Import shared memory (1 page minimum, shared)
  (import "env" "memory" (memory 1 1 shared))

  ;; Constants
  (global $GEN_OFFSET i32 (i32.const 0))    ;; generation at byte 0
  (global $ACTIVE_OFFSET i32 (i32.const 4)) ;; active at byte 4
  (global $SPIN_ITERS i32 (i32.const 100000))

  ;; dispatch_start: called by main thread to kick off a parallel dispatch
  ;; Params: n_threads (i32)
  ;; Returns: new generation value
  (func $dispatch_start (export "dispatch_start") (param $n i32) (result i32)
    (local $new_gen i32)

    ;; Store n_threads in active counter (release)
    (i32.atomic.store
      (global.get $ACTIVE_OFFSET)
      (local.get $n)
    )

    ;; Bump generation with fetch_add (release), get new value
    (local.set $new_gen
      (i32.add
        (i32.atomic.rmw.add
          (global.get $GEN_OFFSET)
          (i32.const 1)
        )
        (i32.const 1)
      )
    )

    ;; Notify all waiters (wake up to 2^32-1 threads)
    (drop
      (memory.atomic.notify
        (global.get $GEN_OFFSET)
        (i32.const 0xffffffff)
      )
    )

    ;; Return new generation
    (local.get $new_gen)
  )

  ;; dispatch_wait: called by main thread to wait for all workers to finish
  ;; Returns: void (spins until active == 0)
  (func $dispatch_wait (export "dispatch_wait")
    (local $spins i32)

    (block $done
      (loop $spin
        ;; Check if active == 0
        (br_if $done
          (i32.eqz
            (i32.atomic.load (global.get $ACTIVE_OFFSET))
          )
        )

        ;; Spin hint (does nothing in WASM but good practice)
        ;; WASM doesn't have a pause instruction, just continue

        ;; Increment spin counter (could add timeout)
        (local.set $spins (i32.add (local.get $spins) (i32.const 1)))

        (br $spin)
      )
    )
  )

  ;; worker_wait: called by worker to wait for new generation
  ;; Params: last_seen_gen (i32)
  ;; Returns: new generation value
  (func $worker_wait (export "worker_wait") (param $last_gen i32) (result i32)
    (local $cur i32)
    (local $spins i32)

    (block $got_new
      (loop $spin
        ;; Load current generation (acquire)
        (local.set $cur
          (i32.atomic.load (global.get $GEN_OFFSET))
        )

        ;; If changed, we're done
        (br_if $got_new
          (i32.ne (local.get $cur) (local.get $last_gen))
        )

        ;; Spin for a bit before futex wait
        (if (i32.lt_u (local.get $spins) (global.get $SPIN_ITERS))
          (then
            (local.set $spins (i32.add (local.get $spins) (i32.const 1)))
            (br $spin)
          )
          (else
            ;; Futex wait: sleep until generation changes
            ;; wait32(addr, expected, timeout_ns) - timeout -1 = infinite
            (drop
              (memory.atomic.wait32
                (global.get $GEN_OFFSET)
                (local.get $last_gen)
                (i64.const -1)
              )
            )
            ;; After wakeup, loop back to recheck
            (br $spin)
          )
        )
      )
    )

    ;; Return new generation
    (local.get $cur)
  )

  ;; worker_done: called by worker when finished with work
  ;; Returns: 1 if this was the last worker, 0 otherwise
  (func $worker_done (export "worker_done") (result i32)
    (local $prev i32)

    ;; Decrement active with fetch_sub (acq_rel)
    (local.set $prev
      (i32.atomic.rmw.sub
        (global.get $ACTIVE_OFFSET)
        (i32.const 1)
      )
    )

    ;; If prev == 1, we were the last one
    (i32.eq (local.get $prev) (i32.const 1))
  )

  ;; get_generation: for testing
  (func $get_generation (export "get_generation") (result i32)
    (i32.atomic.load (global.get $GEN_OFFSET))
  )

  ;; get_active: for testing
  (func $get_active (export "get_active") (result i32)
    (i32.atomic.load (global.get $ACTIVE_OFFSET))
  )

  ;; reset: for benchmarking - reset counters
  (func $reset (export "reset")
    (i32.atomic.store (global.get $GEN_OFFSET) (i32.const 0))
    (i32.atomic.store (global.get $ACTIVE_OFFSET) (i32.const 0))
  )
)
