from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Any


def bootstrap_mixer() -> None:
    try:
        import pygame

        if pygame.mixer.get_init():
            return
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.mixer.init()
    except Exception:
        pass


class AlertPlayer:
    def __init__(self, sound_path: str) -> None:
        self.sound_path = sound_path
        self._sound: Any = None
        self._playing = False
        self._missing_file = False
        self._backend = ""
        self._proc: subprocess.Popen[bytes] | None = None
        self.last_error = ""

    @property
    def backend(self) -> str:
        return self._backend

    def preload(self) -> bool:
        return self._ensure()

    def _ffplay_path(self) -> str | None:
        return shutil.which("ffplay")

    def _use_ffplay(self) -> bool:
        if not sys.platform.startswith("linux"):
            return False
        return self._ffplay_path() is not None

    def _ensure(self) -> bool:
        if self._missing_file:
            return False
        if not os.path.exists(self.sound_path):
            self._missing_file = True
            self.last_error = f"missing file: {self.sound_path}"
            return False
        if self._backend:
            return True

        if self._use_ffplay():
            self._backend = "ffplay"
            self.last_error = ""
            return True

        try:
            import pygame

            bootstrap_mixer()
            if not pygame.mixer.get_init():
                self.last_error = "pygame mixer init failed"
                return False
            pygame.mixer.music.load(self.sound_path)
            pygame.mixer.music.set_volume(1.0)
            self._backend = "pygame.music"
            self.last_error = ""
            return True
        except Exception as exc:
            self.last_error = str(exc)
            return False

    @property
    def is_playing(self) -> bool:
        if not self._playing:
            return False
        if self._backend == "ffplay":
            return self._proc is not None and self._proc.poll() is None
        try:
            import pygame

            return bool(pygame.mixer.music.get_busy())
        except Exception:
            return self._playing

    @property
    def ready(self) -> bool:
        return bool(self._backend) and not self._missing_file

    def pump(self) -> None:
        if not self._playing or self._backend != "pygame.music":
            return
        try:
            import pygame

            pygame.event.pump()
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play(-1)
        except Exception:
            pass

    def _start_ffplay(self, *, loop: bool) -> None:
        ffplay = self._ffplay_path()
        if not ffplay:
            return
        self._stop_ffplay()
        args = [ffplay, "-nodisp", "-loglevel", "quiet"]
        if loop:
            args.extend(["-loop", "0"])
        else:
            args.append("-autoexit")
        args.append(self.sound_path)
        self._proc = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._playing = True

    def _stop_ffplay(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def start(self) -> None:
        if not self._ensure():
            return
        if self._playing and self.is_playing:
            return
        if self._backend == "ffplay":
            self._start_ffplay(loop=True)
            return
        try:
            import pygame

            pygame.mixer.music.play(-1)
            self._playing = True
        except Exception as exc:
            self.last_error = str(exc)

    def stop(self) -> None:
        if self._backend == "ffplay":
            self._stop_ffplay()
        else:
            try:
                import pygame

                pygame.mixer.music.stop()
            except Exception:
                pass
        self._playing = False

    def test(self) -> bool:
        if not self._ensure():
            return False
        if self._backend == "ffplay":
            self._start_ffplay(loop=False)
            self._playing = False
            return True
        try:
            import pygame

            was_looping = self._playing
            if was_looping:
                pygame.mixer.music.stop()
            pygame.mixer.music.play(0)
            self._playing = True
            if was_looping:
                pygame.mixer.music.play(-1)
            return True
        except Exception as exc:
            self.last_error = str(exc)
            return False

    def set_active(self, active: bool) -> None:
        if active:
            self.start()
        else:
            self.stop()
