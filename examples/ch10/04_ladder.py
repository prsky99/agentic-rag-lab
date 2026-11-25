# examples/ch10/04_ladder.py
class Ladder:
    def __init__(self, big, small, cache, breaker):
        self.big, self.small, self.cache, self.breaker = big, small, cache, breaker
    async def generate(self, ctx):
        if not self.breaker.open("big"):
            try:
                return await self.big(ctx), "big"
            except TransientError:
                self.breaker.fail("big")
        if not self.breaker.open("small"):
            try:
                return await self.small(ctx), "small"
            except TransientError:
                self.breaker.fail("small")
        hit = self.cache.lookup(ctx.question, ctx.tenant)
        if hit:
            return hit, "cache"
        raise Abstain("generation_unavailable")
