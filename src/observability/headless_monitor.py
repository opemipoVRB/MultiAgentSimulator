import threading
import queue
import sys
import select


class HeadlessInputMonitor:
    def __init__(self):
        self.input_queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._input_listener, daemon=True)
        self.thread.start()

    def _input_listener(self):
        """Background thread that monitors for keyboard input"""
        while self.running:
            try:
                # Non-blocking check for input
                if select.select([sys.stdin], [], [], 0.1) == ([sys.stdin], [], []):
                    key = sys.stdin.read(1).lower()
                    self.input_queue.put(key)
            except:
                break

    def get_key(self):
        """Get any pending key press"""
        try:
            return self.input_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        """Clean shutdown"""
        self.running = False