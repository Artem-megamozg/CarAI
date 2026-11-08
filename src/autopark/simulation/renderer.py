"""Pygame renderer for the multi-bay AutoPark scene."""

from __future__ import annotations

import math
from typing import Iterable

import pygame

from autopark.environment.parking_env import ParkingEnv
from autopark.physics import CarState, rectangle_corners


class ParkingRenderer:
    WIDTH = 1360
    HEIGHT = 1000
    SCALE = 54
    LEFT = 100
    TOP = 100

    BG = (16, 20, 27)
    ASPHALT = (39, 47, 58)
    ROAD = (45, 54, 66)
    LINE = (165, 177, 191)
    MUTED = (151, 163, 178)
    WHITE = (241, 245, 249)
    GREEN = (62, 215, 151)
    RED = (226, 91, 94)
    YELLOW = (255, 190, 72)
    BLUE = (83, 151, 226)
    TEAL = (68, 177, 195)

    def __init__(self, pygame_module=pygame):
        self.pg = pygame_module
        self.screen = self.pg.display.set_mode((self.WIDTH, self.HEIGHT))
        self.pg.display.set_caption("AutoPark RL v0.3 | Imitation + Reinforcement Learning")
        self.font = self.pg.font.SysFont("segoeui", 21)
        self.small = self.pg.font.SysFont("segoeui", 16)
        self.tiny = self.pg.font.SysFont("segoeui", 13)
        self.large = self.pg.font.SysFont("segoeui", 31, bold=True)

    def world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        return int(self.LEFT + x * self.SCALE), int(self.TOP + y * self.SCALE)

    def _text(self, text: str, x: int, y: int, color=WHITE, font=None) -> None:
        surface = (font or self.font).render(text, True, color)
        self.screen.blit(surface, (x, y))

    def _draw_car(
        self,
        state: CarState,
        length: float,
        width: float,
        color: tuple[int, int, int],
        label: str | None = None,
    ) -> None:
        corners = rectangle_corners(state.x, state.y, state.heading, length, width)
        points = [self.world_to_screen(float(x), float(y)) for x, y in corners]
        self.pg.draw.polygon(self.screen, color, points)
        self.pg.draw.polygon(self.screen, (12, 17, 24), points, 3)

        c, s = math.cos(state.heading), math.sin(state.heading)

        def local_point(local_x: float, local_y: float) -> tuple[int, int]:
            wx = state.x + local_x * c - local_y * s
            wy = state.y + local_x * s + local_y * c
            return self.world_to_screen(wx, wy)

        windshield = [
            local_point(0.18, -width * 0.37),
            local_point(length * 0.22, -width * 0.37),
            local_point(length * 0.22, width * 0.37),
            local_point(0.18, width * 0.37),
        ]
        self.pg.draw.polygon(self.screen, (42, 61, 79), windshield)
        self.pg.draw.polygon(self.screen, (15, 24, 34), windshield, 1)
        self.pg.draw.circle(self.screen, (255, 232, 166), local_point(length * 0.47, 0.0), 5)
        if label:
            label_surface = self.tiny.render(label, True, self.WHITE)
            label_rect = label_surface.get_rect(center=self.world_to_screen(state.x, state.y))
            self.screen.blit(label_surface, label_rect)

    def _draw_bays(self, env: ParkingEnv) -> None:
        selected_id = env.target_slot_id
        for record in env.slot_records:
            x, y = float(record["x"]), float(record["y"])
            center = self.world_to_screen(x, y)
            width = int(env.SLOT_WIDTH * self.SCALE)
            height = int(env.SLOT_LENGTH * self.SCALE)
            rect = self.pg.Rect(0, 0, width, height)
            rect.center = center

            if record["occupied"]:
                fill = (47, 66, 87)
                border = (91, 121, 153)
            elif not record["feasible"]:
                fill = (66, 46, 51)
                border = self.RED
            elif record["id"] == selected_id:
                fill = (76, 91, 47)
                border = self.YELLOW
            else:
                fill = (35, 75, 65)
                border = (73, 164, 128)

            self.pg.draw.rect(self.screen, fill, rect, border_radius=5)
            self.pg.draw.rect(self.screen, border, rect, 3 if record["id"] == selected_id else 2, border_radius=5)
            label_color = self.YELLOW if record["id"] == selected_id else self.MUTED
            self._text(record["id"], rect.left + 7, rect.top + 6, label_color, self.tiny)

            if not record["occupied"] and record["feasible"] and record["id"] != selected_id:
                self.pg.draw.circle(self.screen, self.GREEN, (rect.right - 11, rect.top + 12), 4)

        # Selected spot receives a clear goal marker and arrow for final orientation.
        selected = next(r for r in env.slot_records if r["id"] == selected_id)
        cx, cy = self.world_to_screen(float(selected["x"]), float(selected["y"]))
        direction = -1 if int(selected["row"]) == 0 else 1
        end = (cx, cy + direction * 34)
        self.pg.draw.line(self.screen, self.YELLOW, (cx, cy), end, 4)
        self.pg.draw.circle(self.screen, self.YELLOW, (cx, cy), 7)

    def draw(
        self,
        env: ParkingEnv,
        *,
        episode: int,
        reward: float,
        status: str,
        trajectory: Iterable[tuple[float, float]],
        paused: bool = False,
        deterministic: bool = True,
        mode: str = "PPO POLICY",
    ) -> None:
        self.screen.fill(self.BG)
        self._text("AUTOPARK RL", 72, 25, self.WHITE, self.large)
        self._text(f"MULTI-BAY PARKING  /  {mode}  /  OBSTACLE-AWARE SLOT SELECTION", 75, 65, self.MUTED, self.small)

        arena = self.pg.Rect(
            self.LEFT,
            self.TOP,
            int(env.ARENA_WIDTH * self.SCALE),
            int(env.ARENA_HEIGHT * self.SCALE),
        )
        self.pg.draw.rect(self.screen, self.ASPHALT, arena, border_radius=10)
        self.pg.draw.rect(self.screen, (77, 90, 105), arena, 2, border_radius=10)

        # Driving aisle and dashed centre marking.
        lane_top = self.world_to_screen(0.0, env.LANE_Y - 2.65)[1]
        lane_bottom = self.world_to_screen(0.0, env.LANE_Y + 2.65)[1]
        self.pg.draw.rect(self.screen, self.ROAD, (arena.left + 2, lane_top, arena.width - 4, lane_bottom - lane_top))
        self.pg.draw.line(self.screen, (117, 128, 141), (arena.left + 10, (lane_top + lane_bottom) // 2), (arena.right - 10, (lane_top + lane_bottom) // 2), 2)
        for sx in range(arena.left + 18, arena.right - 15, 55):
            self.pg.draw.line(self.screen, (87, 99, 114), (sx, (lane_top + lane_bottom) // 2), (sx + 25, (lane_top + lane_bottom) // 2), 2)

        # Subtle lot grid.
        for col in range(1, 20):
            sx = self.world_to_screen(float(col), 0.0)[0]
            self.pg.draw.line(self.screen, (46, 54, 65), (sx, arena.top), (sx, arena.bottom), 1)

        self._draw_bays(env)

        for obstacle in env.obstacles:
            self._draw_car(obstacle, env.vehicle.length, env.vehicle.width, self.BLUE)

        path_points = [self.world_to_screen(x, y) for x, y in trajectory]
        if len(path_points) > 1:
            self.pg.draw.lines(self.screen, (255, 190, 72), False, path_points, 3)
        staging = self.world_to_screen(env.staging_x, env.staging_y)
        self.pg.draw.circle(self.screen, (96, 210, 223), staging, 7, 2)
        self.pg.draw.circle(self.screen, (96, 210, 223), staging, 2)
        self._draw_car(env.car, env.vehicle.length, env.vehicle.width, self.YELLOW, "AI")

        # Dashboard panel.
        panel_y = arena.bottom + 18
        free = sum(not bool(r["occupied"]) for r in env.slot_records)
        feasible = sum(bool(r["feasible"]) for r in env.slot_records)
        self._text(f"EPISODE  {episode}", 76, panel_y, self.WHITE)
        self._text(f"TARGET  {env.target_slot_id}", 270, panel_y, self.YELLOW)
        self._text(f"FREE BAYS  {free}/12", 460, panel_y, self.GREEN)
        self._text(f"FEASIBLE  {feasible}", 650, panel_y, self.TEAL)
        self._text(f"REWARD  {reward:8.1f}", 850, panel_y, self.WHITE)
        self._text(f"SPEED  {env.car.speed:+.2f} m/s", 1080, panel_y, self.WHITE)

        phase = "APPROACH + ALIGN" if env.stage == 0 else "ENTER BAY + STOP"
        self._text(f"PHASE  {phase}", 76, panel_y + 34, self.TEAL if env.stage == 0 else self.YELLOW)
        self._text(f"DISTANCE  {env._target_distance():.2f} m", 350, panel_y + 34, self.WHITE)
        self._text(f"ANGLE ERROR  {math.degrees(abs(env._angle_error())):.1f} deg", 580, panel_y + 34, self.WHITE)
        self._text(f"STEPS  {env.step_count}/{env.max_steps}", 900, panel_y + 34, self.WHITE)

        status_color = self.GREEN if "SUCCESS" in status else self.RED if ("COLLISION" in status or "OUTSIDE" in status) else self.YELLOW
        self._text(status, 76, panel_y + 68, status_color, self.font)
        mode = "DETERMINISTIC" if deterministic else "STOCHASTIC"
        self._text(f"POLICY  {mode}     SPACE pause  |  N next episode  |  R reset  |  ESC exit", 76, self.HEIGHT - 29, self.MUTED, self.small)

        if paused:
            overlay = self.pg.Surface((self.WIDTH, self.HEIGHT), self.pg.SRCALPHA)
            overlay.fill((5, 8, 12, 135))
            self.screen.blit(overlay, (0, 0))
            pause_surface = self.large.render("PAUSED", True, self.WHITE)
            self.screen.blit(pause_surface, pause_surface.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2)))

        self.pg.display.flip()
