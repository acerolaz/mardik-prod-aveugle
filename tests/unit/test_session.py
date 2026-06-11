import threading

from mardik.session import SessionStore


def test_record_turn_counts_every_concurrent_turn():
    store = SessionStore()
    session_id = "load-test"
    workers = 20
    per_worker = 50

    def hammer() -> None:
        for _ in range(per_worker):
            store.record_turn(session_id)

    threads = [threading.Thread(target=hammer) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert store.turns(session_id) == workers * per_worker
