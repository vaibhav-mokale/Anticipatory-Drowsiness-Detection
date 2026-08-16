from __future__ import annotations

import os
from typing import Any


class AlertPlayer:
    def __init__(self, sound_path: str) -> None:
        self.sound_path = sound_path
        self._sound: Any = None
        self._mixer: Any = None
        self._playing = False
        self._failed = False
        self.last_error = ""

    def preload(self) -> bool:
        return self._ensure()

    def _ensure(self) -> bool:
        if self._failed:
            return False
        if self._sound is not None:
            return True
        if not os.path.exists(self.sound_path):
            self._failed = True
            self.last_error = f"missing file: {self.sound_path}"
            return False
        try:
            import pygame

            if not pygame.mixer.get_init():
                pygame.mixer.init(
                    frequency=44100,
                    size=-16,
                    channels=2,
                    buffer=2048,
                )
            self._sound = pygame.mixer.Sound(self.sound_path)
            self._sound.set_volume(1.0)
            self._mixer = pygame.mixer
            return True
        except Exception as exc:
            self._failed = True
            self.last_error = str(exc)
            return False

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def ready(self) -> bool:
        return self._sound is not None and not self._failed

    def start(self) -> None:
        if self._playing:
            return
        if not self._ensure():
            return
        self._sound.play(loops=-1)
        self._playing = True

    def stop(self) -> None:
        if not self._playing:
            return
        if self._sound is not None:
            self._sound.stop()
        self._playing = False

    def set_active(self, active: bool) -> None:
        if active:
            self.start()
        else:
            self.stop()
