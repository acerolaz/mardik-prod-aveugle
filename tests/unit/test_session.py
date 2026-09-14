import threading

from mardik.session import SessionStore


def test_commit_turn_counts_every_concurrent_turn():
    store = SessionStore()
    session_id = "load-test"
    workers = 20
    per_worker = 50
    user = {"role": "user", "content": "q"}
    assistant = {"role": "assistant", "content": "a"}

    def hammer() -> None:
        for _ in range(per_worker):
            store.commit_turn(session_id, user, assistant)

    threads = [threading.Thread(target=hammer) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert store.turns(session_id) == workers * per_worker
    assert store.history(session_id) == [user, assistant] * (workers * per_worker)
