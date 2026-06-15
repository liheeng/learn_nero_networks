from manim import *
import numpy as np

from mlp_layers import SimpleMLP
from utils import make_two_clusters


class MLPFlow(Scene):
    def construct(self):

        # ===== 1. data =====
        X, labels = make_two_clusters()

        mlp = SimpleMLP()
        stages = mlp.forward_stage(X)

        colors = [BLUE, RED]

        def make_dots(points):
            dots = VGroup()
            for i, p in enumerate(points):
                dot = Dot(
                    point=np.array([p[0], p[1], 0]),
                    radius=0.04,
                    color=colors[labels[i]],
                )
                dots.add(dot)
            return dots

        # ===== 2. stage 0 =====
        dots = make_dots(stages[0])
        self.play(FadeIn(dots))
        self.wait(1)

        title = Text("Input Space").to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        # ===== 3. animate through layers =====
        stage_names = ["Linear + ReLU (Layer 1)", "Linear + ReLU (Layer 2)"]

        for i in range(1, len(stages)):

            new_dots = make_dots(stages[i])

            new_title = Text(stage_names[i - 1]).to_edge(UP)

            self.play(
                Transform(dots, new_dots), Transform(title, new_title), run_time=2
            )

            self.wait(1)

        # ===== 4. final emphasis =====
        self.play(dots.animate.scale(1.2), run_time=1)

        self.wait(2)
