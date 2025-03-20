class STATES:
    defualt = 0
    damp = 1
    ctrl = 2


class stateMachine:
    def __init__(self):
        self.state = STATES.defualt

    def Stop(self):
        self.state = STATES.damp
        return True

    def Defualt(self):
        if self.state == STATES.damp or self.state == STATES.ctrl:
            self.state = STATES.defualt
            return True
        else:
            return False

    def Ctrl(self):
        if self.state == STATES.damp or self.state == STATES.defualt:
            self.state = STATES.ctrl
            return True
        else:
            return False

