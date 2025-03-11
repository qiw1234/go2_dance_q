class STATES:
    stand = 0
    damp = 1
    ctrl = 2


class stateMachine:
    def __init__(self):
        self.state = STATES.stand

    def Stop(self):
        self.state = STATES.damp
        return True

    def Stand(self):
        if self.state == STATES.damp or self.state == STATES.ctrl:
            self.state = STATES.stand
            return True
        else:
            return False

    def Ctrl(self):
        if self.state == STATES.damp or self.state == STATES.stand:
            self.state = STATES.ctrl
            return True
        else:
            return False

