from pathlib import Path

import numpy as np
import trimesh
import tyro
import viser
import viser.transforms as tf
import time
import pandas as pd

from scipy.spatial.transform import Rotation as R
import mediapy

import matplotlib as mpl
n_colors = 6 # not really but we want to avoid the light yellows
cmap = mpl.colormaps['plasma']
colors = {
    '15':cmap(np.linspace(0, 1, n_colors))[0],
    '30':cmap(np.linspace(0, 1, n_colors))[1],
    '45':cmap(np.linspace(0, 1, n_colors))[2],
    '60':cmap(np.linspace(0, 1, n_colors))[3],
    '90':cmap(np.linspace(0, 1, n_colors))[4],
    }

def get_pos_rot(theta_, phi_, psi_):
    theta = theta_ / 180 * np.pi
    phi = phi_ / 180 * np.pi
    psi = psi_ / 180 * np.pi

    x, y, z = np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)
    position = np.array([x, y, z])

    roll = tf.SO3.from_z_radians(psi)
    dir  = tf.SO3.multiply(tf.SO3.from_z_radians(phi + np.pi/2), tf.SO3.from_x_radians(np.pi + theta))
    rotation = tf.SO3.multiply(dir, roll).wxyz

    return position, rotation


def main(
        data: Path,
        port: int = 7018,
    ):
    mesh = trimesh.load_mesh(Path(data))
    name = str(data).split(".")[-2].split("/")[-1]
    output_dir = Path(f"data/{name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # center and normalize
    mesh.vertices -= np.mean(mesh.vertices, axis=0)
    mesh.vertices /= np.max(np.abs(mesh.vertices))

    server = viser.ViserServer(port=port)
    # server.request_share_url()

    grid_button = server.gui.add_button("Render Uniform Samples")
    rand_button = server.gui.add_button("Render Random Samples")

    @grid_button.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None

        renders_dir = output_dir / "renders"
        renders_dir.mkdir(parents=True,exist_ok=True)

        res = 256

        print("Generating images")
        
        r = 2
        step = 10
        for step in [15, 30, 45, 60, 90]:
            thetas = np.arange(0, 180 + step, step)
            psis = np.arange(0, 360, step) # roll

            height, width = res, res
            f = 1

            ttl_phis = 0
            for i, theta_ in enumerate(thetas):
                num_phis = round(360 / step * np.sin(theta_ * np.pi / 180))
                num_phis = int(max(num_phis, 1))
                phis = np.linspace(0, 360, num_phis + 1)[:-1]

                ttl_phis += len(phis)
                # print(len(phis))
                for j, phi_ in enumerate(phis):
                    for k, psi_ in enumerate(psis):
                        if (theta_ == 0 or theta_ == 180) and phi_ > 0:
                            continue

                        position, rotation = get_pos_rot(theta_, phi_, psi_)

                        client.camera.position = r * position
                        client.camera.wxyz = rotation
                        image = client.camera.get_render(height=res, width=res, transport_format='png')

                        # if k == 0:
                        #     grid[i * res: (i+1) * res, j*res:(j+1)*res] = image
                        mediapy.write_image(renders_dir / f"render_{theta_:06.2f}_{phi_:06.2f}_{psi_:06.2f}.png", image)
        # mediapy.write_image(f"renders/grid.png", grid)

        print("Done!")

    @rand_button.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None

        renders_dir = output_dir / "rand_renders"
        renders_dir.mkdir(parents=True,exist_ok=True)

        print("Generating images")

        n = 1000
        r = 2
        res = 256
        # uniform random sampling https://www.jasondavies.com/maps/random-points/
        thetas = np.round(np.arccos(2*np.random.rand(n) - 1) / np.pi * 180, 2)
        phis = np.round(np.random.rand(n) * 360, 2)
        psis = np.round(np.random.rand(n) * 360, 2)

        for theta_, phi_, psi_ in zip(thetas, phis, psis):
            position, rotation = get_pos_rot(theta_, phi_, psi_)

            client.camera.position = r * position
            client.camera.wxyz = rotation
            image = client.camera.get_render(height=res, width=res, transport_format='png')


            mediapy.write_image(renders_dir / f"render_{theta_}_{phi_}_{psi_}.png", image)

        print("Done!")

    server.scene.add_mesh_simple(
            name="/mesh",
            vertices=mesh.vertices,
            faces=mesh.faces,
            color=(255,255,255),
            flat_shading=True,
            side="double",
            wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
            position=(0.0, 0.0, 0.0),
        )

    def draw(step=30):

        loss = "dino"
        df = pd.read_csv(f"data/viewpoint_est/{name}/{loss}_{step}.csv")
        # ref = df.iloc[687]["reference"][:-4]
        # match = df.iloc[687]["best_match"][:-4]

        # ref = df.iloc[605]["reference"][:-4]
        # match = df.iloc[605]["best_match"][:-4]

        ref = df.iloc[456]["reference"][:-4]
        match = df.iloc[456]["best_match"][:-4]

        print(ref, match)

        ref_theta, ref_phi, ref_psi = ref.split("_")
        match_theta, match_phi, match_psi = match.split("_")

        r = 2
        thetas = np.arange(0, 180 + step, step)
        psis = np.arange(0, 360, step) # roll

        height, width = 256, 256
        f = 1

        scale = 0.08

        ref_position, ref_rotation = get_pos_rot(float(ref_theta), float(ref_phi), float(ref_psi))
        server.scene.add_camera_frustum(
                    f"consistent_{step}/camera_{ref_theta}_{ref_phi}",
                    fov=np.arctan2(height / 2, f),
                    aspect=width / height,
                    scale=scale,
                    wxyz= ref_rotation,
                    position= r * ref_position,
                )

        ttl_phis = 0
        # print(thetas)
        for i, theta_ in enumerate(thetas):
            num_phis = round(360 / step * np.sin(theta_ * np.pi / 180))
            num_phis = int(max(num_phis, 1))
            phis = np.linspace(0, 360, num_phis + 1)[:-1]
            ttl_phis += len(phis)
            # print(len(phis))
            # print(f"{theta_}: {180 / step * np.sin(i * step * np.pi / 180)}")
            
            for j, phi_ in enumerate(phis):
                psi_ = 0
                if theta_ == 0 or theta_ == 180:
                    psi_ = 0
                    phi_ = 0

                position, rotation = get_pos_rot(theta_, phi_, psi_)

                # server.scene.add_camera_frustum(
                #     f"consistent_{step}/camera_{theta_}_{phi_}",
                #     fov=np.arctan2(height / 2, f),
                #     aspect=width / height,
                #     scale=0.05,
                #     wxyz= rotation,
                #     position= r * position,
                # )
                # psis = np.arange(0, 360, step)[:-1]
                psis = [0]
                for k, psi_ in enumerate(psis):
                    position, rotation = get_pos_rot(theta_, phi_, psi_)
                    
                    color = (100, 100, 100)
                    scale = 0.075
                    # color = (0, 0, 0)
                    if f"{theta_:06.2f}" == match_theta and f"{phi_:06.2f}" == match_phi and f"{psi_:06.2f}" == match_psi:
                        # color = (106, 0, 167)
                        color = colors[str(step)][:3]
                        scale = 0.08
                    server.scene.add_camera_frustum(
                        f"consistent_{step}/camera_{theta_}_{phi_}_{psi_}",
                        fov=np.arctan2(height / 2, f),
                        aspect=width / height,
                        scale=scale,
                        color=color,
                        wxyz= rotation,
                        position= (2 + (k * 0)) * position,
                    )

        print(ttl_phis * (360//step))

    def draw2():
        r = 2
        res = 256
        step = 60
        thetas = np.arange(5, 180 + step//2, step//2)
        phis = np.arange(0, 360, step)
        psis = np.arange(0, 360, step) # roll

        height, width = 256, 256
        f = 1

        for i, theta_ in enumerate(thetas):
            for j, phi_ in enumerate(phis):
                if theta_ == 0 or theta_ == 180:
                    psi_ = 0
                    phi_ = 0

                position, rotation = get_pos_rot(theta_, phi_, 0)

                server.scene.add_camera_frustum(
                    f"naive/camera_{theta_}_{phi_}",
                    fov=np.arctan2(height / 2, f),
                    aspect=width / height,
                    scale=0.05,
                    wxyz= rotation,
                    position= r * position,
                )

        print(len(thetas) * len(phis)  * (360//step)) 
        
        
    # draw(step=15)
    draw(step=30)
    # draw(step=45)
    # draw(step=60)
    # draw(step=90)
    # draw2()


    # import pdb; pdb.set_trace()
    while True:
        # gui_theta.on_update(lambda _: draw())
        # gui_phi.on_update(lambda _: draw())
        # gui_rho.on_update(lambda _: draw())
        time.sleep(1.0 / 30)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(main)


if __name__ == "__main__":
    entrypoint()