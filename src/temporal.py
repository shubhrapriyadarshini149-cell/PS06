from collections import deque
from typing import List

class TemporalSmoother:
    def __init__(self, n_frames=15, m_frames=10, max_age=30):
        self.N = n_frames
        self.M = m_frames
        self.max_age = max_age
        
        # track_id -> deque of lists of missing PPE items
        self.history = {}
        # track_id -> set of currently confirmed missing PPE items
        self.state = {}
        # track_id -> last seen frame count
        self.last_seen = {}

    def update(self, track_id: int, missing: list, frame_count: int) -> list:
        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=self.N)
            self.state[track_id] = set()
            
        self.history[track_id].append(missing)
        self.last_seen[track_id] = frame_count
        
        # Count missing occurrences in the last N frames
        counts = {}
        for missing_list in self.history[track_id]:
            for item in missing_list:
                counts[item] = counts.get(item, 0) + 1
                
        # Update confirmed state based on M-of-N
        for item in list(self.state[track_id]):
            # Needs M compliant (not missing) frames to recover
            not_missing = len(self.history[track_id]) - counts.get(item, 0)
            if not_missing >= self.M:
                self.state[track_id].remove(item)
                
        for item, count in counts.items():
            if count >= self.M:
                self.state[track_id].add(item)
                
        return list(self.state[track_id])

    def cleanup(self, current_frame: int):
        to_delete = []
        for tid, last_f in self.last_seen.items():
            if current_frame - last_f > self.max_age:
                to_delete.append(tid)
                
        for tid in to_delete:
            del self.history[tid]
            del self.state[tid]
            del self.last_seen[tid]
