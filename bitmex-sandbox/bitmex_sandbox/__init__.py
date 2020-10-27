from gym.envs.registration import register

register(
    id='bitbox-v0',
    entry_point='bitmex_sandbox.envs:SandboxSnd',
)