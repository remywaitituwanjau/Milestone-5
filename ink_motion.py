"""
Ink & Motion -- Python port
Milestone 5: Dynamics & Animation, applied to Non-Photorealistic Rendering (NPR).

A physics-driven bouncing figure with:
  - real gravity/restitution integration (not keyframed)
  - a damped spring layer producing volume-conserving squash & stretch
  - an ink-droplet particle system spawned on ground contact (gravity + drag + fade)
  - live tracking of total mechanical energy, used afterwards for the
    stability / realism evaluation

Outputs:
  ink_motion.gif           -- the animated system
  motion_analysis.png      -- energy-over-time + peak-ratio stability plot
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

# ---------------------------------------------------------------- parameters
GRAVITY      = 650.0   # px/s^2
RESTITUTION  = 0.72    # energy retained (velocity fraction) per bounce
SQUASH_AMT   = 0.55    # how strongly impact speed drives squash
SPRING_K     = 90.0    # squash spring stiffness
SPRING_DAMP  = 10.0    # squash spring damping
BOIL_AMT     = 1.2     # hand-drawn outline jitter strength

W, H = 640, 420
GROUND_Y = H - 60
R = 34.0               # base radius of the figure
FPS = 30
DURATION_S = 7.0
N_FRAMES = int(FPS * DURATION_S)
DT = 1.0 / FPS

INK   = "#242220"
PAPER = "#efe6d0"
JADE  = "#4e6e58"
SEAL  = "#a5342a"

rng = np.random.default_rng(7)

# ---------------------------------------------------------------- state
class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "r", "drag")

    def __init__(self, x, y, speed):
        ang = -np.pi/2 + (rng.random()-0.5)*1.9
        s = speed * (0.25 + rng.random()*0.5)
        self.x, self.y = x, y
        self.vx = np.cos(ang)*s*0.6 + (rng.random()-0.5)*40
        self.vy = np.sin(ang)*s*0.6 - rng.random()*60
        self.life = 0.0
        self.max_life = 0.5 + rng.random()*0.6
        self.r = 1.2 + rng.random()*2.4
        self.drag = 0.6 + rng.random()*0.5

    def step(self, dt):
        self.life += dt
        self.vy += GRAVITY*dt
        self.vx *= (1 - self.drag*dt)
        self.vy *= (1 - self.drag*0.3*dt)
        self.x += self.vx*dt
        self.y += self.vy*dt

    def alive(self):
        return self.life < self.max_life and self.y < GROUND_Y + 2


class Sim:
    def __init__(self):
        self.x, self.y = 140.0, -260.0    # y measured upward from ground (0 = contact)
        self.vx, self.vy = 60.0, 0.0
        self.scaleX, self.scaleY = 1.0, 1.0
        self.scaleVY = 0.0
        self.particles = []
        self.stains = []          # permanent ink stains: (x, r, alpha)
        self.energy_hist = []
        self.peak_heights = []
        self.current_apex = 0.0
        self.bounces = 0
        self.t = 0.0
        self.settled = False

    def energy(self):
        h = -self.y
        ke = 0.5*1.0*self.vy**2
        pe = 1.0*GRAVITY*max(h, 0)
        return (ke + pe) * 1e-4   # scaled to friendly units

    def step(self, dt):
        self.t += dt

        if not self.settled:
            self.vy += GRAVITY*dt
            self.y += self.vy*dt
            self.x += self.vx*dt
            if self.x < R+10 or self.x > W-R-10:
                self.vx *= -1
                self.x = np.clip(self.x, R+10, W-R-10)

            if self.y < self.current_apex:
                self.current_apex = self.y

            if self.y >= 0:
                impact_speed = abs(self.vy)
                self.y = 0.0
                self.bounces += 1
                apex_h = -self.current_apex
                if apex_h > 1:
                    self.peak_heights.append(apex_h)
                self.current_apex = 0.0

                sq = min(0.85, (impact_speed/380.0)*SQUASH_AMT*2.2)
                self.scaleVY -= sq*10
                self._spawn_burst(self.x, impact_speed)

                if impact_speed < 25:
                    # below this threshold the bounce no longer clears a visible
                    # height -- treat the system as having converged to rest
                    self.vy = 0.0
                    self.settled = True
                else:
                    self.vy = -impact_speed * RESTITUTION

        # squash/stretch spring, volume-conserving (keeps running briefly after rest)
        self.scaleVY += (-(self.scaleY-1)*SPRING_K)*dt
        self.scaleVY *= (1 - SPRING_DAMP*dt)
        self.scaleY += self.scaleVY*dt
        self.scaleY = np.clip(self.scaleY, 0.35, 1.6)
        self.scaleX = 1.0/np.sqrt(self.scaleY)

        # particles
        alive = []
        for p in self.particles:
            p.step(dt)
            if p.alive():
                alive.append(p)
            else:
                self.stains.append((p.x, p.r*1.4, 0.16))
        self.particles = alive

        self.energy_hist.append(self.energy())

    def _spawn_burst(self, x, speed):
        n = min(22, 6 + int(speed/18))
        for _ in range(n):
            self.particles.append(Particle(x, 0.0, speed))


# ---------------------------------------------------------------- run sim headless first (for reproducibility of the analysis plot)
sim = Sim()
for _ in range(N_FRAMES):
    sim.step(DT)

# ---------------------------------------------------------------- figure export (re-run sim in sync with rendering)
fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=110)
fig.patch.set_facecolor(PAPER)


def render():
    sim2 = Sim()

    def fy(y):
        # flip screen-space y (0 at top) into matplotlib's bottom-up axes,
        # avoiding ax.invert_yaxis() which breaks Polygon clip paths
        return H - y

    def draw(frame):
        ax.clear()
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_facecolor(PAPER)
        ax.axis("off")

        sim2.step(DT)

        # ground line
        gx = np.linspace(0, W, 120)
        gy = GROUND_Y + np.sin(gx*0.05 + sim2.t*0.4)*0.6
        ax.plot(gx, fy(gy), color=INK, lw=1.4, alpha=0.5)

        # permanent ink stains
        for (sx, sr, sa) in sim2.stains[-400:]:
            ax.add_patch(patches.Ellipse((sx, fy(GROUND_Y-1)), sr*2.6, sr*1.1,
                                          facecolor=INK, alpha=sa, lw=0))

        cx, cy = sim2.x, fy(GROUND_Y + sim2.y)
        rx, ry = R*sim2.scaleX, R*sim2.scaleY

        # contact shadow
        shadow_scale = max(0.35, 1 - (-sim2.y)/220)
        ax.add_patch(patches.Ellipse((cx, fy(GROUND_Y+4)), R*2.1*shadow_scale, R*0.56*shadow_scale,
                                      facecolor=INK, alpha=0.18*shadow_scale, lw=0))

        # boiling outline (hand-drawn jitter)
        n = 40
        angs = np.linspace(0, 2*np.pi, n, endpoint=False)
        jitter = (np.sin(angs*3 + sim2.t*3.1)*0.02*BOIL_AMT +
                  np.sin(angs*5 - sim2.t*2.3)*0.015*BOIL_AMT)
        px = cx + np.cos(angs)*rx*(1+jitter)
        py = cy + np.sin(angs)*ry*(1+jitter*0.8)
        ax.fill(px, py, color=PAPER, ec=INK, lw=2.0, zorder=5)

        # toon shadow crescent + hatching, clipped to figure
        clip_patch = patches.Polygon(np.column_stack([px, py]), closed=True, transform=ax.transData)
        crescent = patches.Ellipse((cx+rx*0.42, cy-ry*0.32), rx*1.7, ry*1.7,
                                    facecolor=JADE, alpha=0.35, lw=0, zorder=6)
        crescent.set_clip_path(clip_patch)
        ax.add_patch(crescent)

        for d in np.arange(-rx*2, rx*2, 5):
            l, = ax.plot([cx-rx*2+d, cx-rx*2+d+ry*2],
                         [cy-ry*2, cy+ry*2], color=INK, lw=0.6, alpha=0.35, zorder=6)
            l.set_clip_path(clip_patch)

        # eye
        ax.add_patch(patches.Ellipse((cx-rx*0.18, cy+ry*0.15), 4.5, 5.5*sim2.scaleY,
                                      facecolor=INK, lw=0, zorder=7))

        # particles
        for p in sim2.particles:
            a = max(0, 1 - p.life/p.max_life)
            ax.add_patch(patches.Circle((p.x, fy(GROUND_Y+p.y)), p.r,
                                         facecolor="#3a352f", alpha=a*0.85, lw=0, zorder=8))

        # HUD text
        ax.text(10, H-22, f"bounce {sim2.bounces}   speed {abs(sim2.vy)*0.05:.1f}   "
                           f"energy {sim2.energy():.2f} J",
                fontsize=8, color="#514c43", family="monospace")

        return []

    anim = FuncAnimation(fig, draw, frames=N_FRAMES, blit=False, interval=1000/FPS)
    anim.save("/home/claude/ink_motion.gif", writer=PillowWriter(fps=FPS))


render()
plt.close(fig)

# ---------------------------------------------------------------- motion analysis / stability evaluation figure
fig2, axs = plt.subplots(1, 2, figsize=(10, 3.6), dpi=130)
fig2.patch.set_facecolor("white")

t_axis = np.arange(len(sim.energy_hist))*DT
axs[0].plot(t_axis, sim.energy_hist, color=SEAL, lw=1.4)
axs[0].set_title("Total mechanical energy vs. time", fontsize=11)
axs[0].set_xlabel("time (s)")
axs[0].set_ylabel("energy (arb. units)")
axs[0].grid(alpha=0.25)

peaks = np.array(sim.peak_heights)
ratios = peaks[1:] / peaks[:-1] if len(peaks) > 1 else np.array([])
expected = RESTITUTION**2
axs[1].plot(range(1, len(ratios)+1), ratios, "o-", color=JADE, label="measured ratio")
axs[1].axhline(expected, color=SEAL, ls="--", lw=1.2, label=f"expected e\u00b2 = {expected:.3f}")
axs[1].set_title("Bounce peak-to-peak ratio (stability check)", fontsize=11)
axs[1].set_xlabel("bounce index")
axs[1].set_ylabel("apex[i] / apex[i-1]")
axs[1].set_ylim(0, 1)
axs[1].legend(fontsize=8)
axs[1].grid(alpha=0.25)

plt.tight_layout()
plt.savefig("/home/claude/motion_analysis.png")
plt.close(fig2)

print("bounces:", sim.bounces)
print("peak heights:", sim.peak_heights[:6], "...")
print("expected ratio (e^2):", expected)
if len(ratios):
    print("measured ratios:", np.round(ratios, 3))
    print("mean abs deviation from expected:", np.mean(np.abs(ratios-expected)))
print("done")
