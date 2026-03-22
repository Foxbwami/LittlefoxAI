class ChatMemory:
    def __init__(self, max_tokens=100):
        self.history = []
        self.max_tokens = max_tokens

    def add(self, role, message):
        self.history.append((role, message))

    def last_user_message(self, skip_latest=False):
        items = self.history[:-1] if skip_latest else self.history
        for role, msg in reversed(items):
            if role.lower() == "user":
                return msg
        return ""

    def build_context(self):
        context = ""
        total_tokens = 0

        for role, msg in reversed(self.history):
            tokens = msg.split()
            total_tokens += len(tokens)

            if total_tokens > self.max_tokens:
                break

            context = f"{role}: {msg} " + context

        return context.strip()
