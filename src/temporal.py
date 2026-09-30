from collections import deque

class TemporalSmoother:
    def __init__(self, history_size=30, violation_threshold=0.6, compliance_threshold=0.8):
        self.history_size = history_size
        self.violation_threshold = violation_threshold
        self.compliance_threshold = compliance_threshold
        
        # track_id -> deque of bools (True for violation, False for compliant)
        self.history = {}
        # track_id -> current steady state
        self.state = {}

    def update(self, track_id: int, is_violation: bool) -> str:
        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=self.history_size)
            self.state[track_id] = "COMPLIANT" # default start
            
        self.history[track_id].append(is_violation)
        
        history_list = list(self.history[track_id])
        violation_ratio = sum(history_list) / len(history_list)
        
        if self.state[track_id] == "COMPLIANT":
            if violation_ratio >= self.violation_threshold:
                self.state[track_id] = "VIOLATION"
        else: # currently VIOLATION
            # need a high compliance ratio (low violation ratio) to recover
            if (1.0 - violation_ratio) >= self.compliance_threshold:
                self.state[track_id] = "COMPLIANT"
                
        return self.state[track_id]
