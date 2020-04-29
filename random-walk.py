# Andrea Rahmadanisya 1301184146
# Vijay Cheza Pangestu 1301180351

import numpy as np
#-------------------- animation -------------------
import pygame
import sys
#---------------------------------------------------
# Inisilaisasi untuk menggunakan variabel yang di assign dengan komposisi warna
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (119, 158, 203)
RED = (255, 105, 97)
GREEN = (190, 199, 180)
YELLOW = (190, 199, 180)
GREY = (230, 230, 230)
BACKGROUND = WHITE

# Merupakan kelas object atau individu yang bergerak
class individu(pygame.sprite.Sprite):
    def __init__(
        self,
        x,
        y,
        width,
        height,
        color=BLACK,
        radius=5,
        velocity=[0, 0],
        randomize=False,
    ):
        super().__init__()
        self.image = pygame.Surface([radius * 2, radius * 2])
        self.image.fill(BACKGROUND)
        pygame.draw.circle(
            self.image, color, (radius, radius), radius
        )

        self.rect = self.image.get_rect()
        # Inisialisasi container untuk mendapatkan nilai x,y yang
        # nanti dibangkitkan dari bilangan acak
        self.pos = np.array([x, y], dtype=np.float64)
        # menggunakan velocity sebagai kecepatan dan juga arah
        # dalam bergerak pada library pygame
        self.vel = np.asarray(velocity, dtype=np.float64)

        self.killswitch_on = False
        self.recovered = False
        self.randomize = randomize

        self.WIDTH = width
        self.HEIGHT = height

    def update(self):

        self.pos += self.vel

        x, y = self.pos

        # Periodic boundary conditions (PBC)
        if x < 0:
            self.pos[0] = self.WIDTH
            x = self.WIDTH
        if x > self.WIDTH:
            self.pos[0] = 0
            x = 0
        if y < 0:
            self.pos[1] = self.HEIGHT
            y = self.HEIGHT
        if y > self.HEIGHT:
            self.pos[1] = 0
            y = 0

        self.rect.x = x
        self.rect.y = y

        vel_norm = np.linalg.norm(self.vel)
        if vel_norm > 3:
            self.vel /= vel_norm

        if self.randomize:
            # Yaitu dengan membangkitkan angka random dari [-1,1]
            self.vel += np.random.rand(2) * 2 - 1

        if self.killswitch_on:
            self.waktu_pemulihan -= 1

            if self.waktu_pemulihan <= 0:
                self.killswitch_on = False
                some_number = np.random.rand()
                if self.imun_recovery > some_number:
                    self.kill()
                else:
                    self.recovered = True

    def respawn(self, color, radius=5):
        return individu(
            self.rect.x,
            self.rect.y,
            self.WIDTH,
            self.HEIGHT,
            color=color,
            velocity=self.vel,
        )

    def killswitch(self, waktu_pemulihan=10, imun_recovery=0.2):
        self.killswitch_on = True
        self.waktu_pemulihan = waktu_pemulihan
        self.imun_recovery = imun_recovery


# Merupakan kelas untuk simulasi dapat berjalan
class Simulation:
    def __init__(self, width=756, height=756):
        self.WIDTH = width
        self.HEIGHT = height

        self.individu_container = pygame.sprite.Group()
        self.terinfeksi_container = pygame.sprite.Group()
        self.recovered_container = pygame.sprite.Group()
        self.all_container = pygame.sprite.Group()

        # Inisialisasi variabel skalar secara default
        self.jumlah_individu = 200
        self.individu_terinfeksi = 10
        self.probabilitas_Tidakbergerak = 40
        self.T = 500  # Merupakan Time Stamp
        self.waktu_pemulihan = 10
        self.imun_recovery = 0.2

    def start(self, randomize=False):

        self.N = (
            self.jumlah_individu + self.individu_terinfeksi + self.probabilitas_Tidakbergerak
        )

        pygame.init()
        screen = pygame.display.set_mode([self.WIDTH, self.HEIGHT])

        for i in range(self.jumlah_individu):
            # Dibangkitkan dari nilai random
            x = np.random.randint(0, self.WIDTH + 1)
            # Dibangkitkan dari nilai random
            y = np.random.randint(0, self.HEIGHT + 1)
            # Dibangkitkan dari nilai random [-1,1]
            vel = np.random.rand(2) * 2 - 1
            guy = individu(
                x,
                y,
                self.WIDTH,
                self.HEIGHT,
                color=BLUE,
                velocity=vel,
                randomize=randomize,
            )
            self.individu_container.add(guy)
            self.all_container.add(guy)

        for i in range(self.probabilitas_Tidakbergerak):
            x = np.random.randint(0, self.WIDTH + 1)
            y = np.random.randint(0, self.HEIGHT + 1)
            vel = [0, 0]
            guy = individu(
                x,
                y,
                self.WIDTH,
                self.HEIGHT,
                color=BLUE,
                velocity=vel,
                randomize=False,
            )
            self.individu_container.add(guy)
            self.all_container.add(guy)

        for i in range(self.individu_terinfeksi):
            x = np.random.randint(0, self.WIDTH + 1)
            y = np.random.randint(0, self.HEIGHT + 1)
            vel = np.random.rand(2) * 2 - 1
            guy = individu(
                x,
                y,
                self.WIDTH,
                self.HEIGHT,
                color=RED,
                velocity=vel,
                randomize=randomize,
            )
            self.terinfeksi_container.add(guy)
            self.all_container.add(guy)

        stats = pygame.Surface((self.WIDTH // 4, self.HEIGHT // 4))
        stats.fill(GREY)
        stats.set_alpha(230)
        stats_pos = (self.WIDTH // 40, self.HEIGHT // 40)

        clock = pygame.time.Clock()

        for i in range(self.T):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

            self.all_container.update()

            screen.fill(BACKGROUND)

            # Statistik Grafik Penyebaran
            stats_height = stats.get_height()
            stats_width = stats.get_width()
            n_inf_now = len(self.terinfeksi_container)
            n_pop_now = len(self.all_container)
            n_rec_now = len(self.recovered_container)
            t = int((i / self.T) * stats_width)
            y_infect = int(
                stats_height - (n_inf_now / n_pop_now) * stats_height
            )
            y_dead = int(
                ((self.N - n_pop_now) / self.N) * stats_height
            )
            y_recovered = int((n_rec_now / n_pop_now) * stats_height)
            stats_graph = pygame.PixelArray(stats)
            stats_graph[t, y_infect:] = pygame.Color(*RED)
            stats_graph[t, :y_dead] = pygame.Color(*YELLOW)
            stats_graph[
                t, y_dead: y_dead + y_recovered
            ] = pygame.Color(*GREEN)

            # Individu yang menginfeksi (individu - infection)
            collision_group = pygame.sprite.groupcollide(
                self.individu_container,
                self.terinfeksi_container,
                True,
                False,
            )

            for guy in collision_group:
                new_guy = guy.respawn(RED)
                # kecepatan & arah akan berlawanan jika menyentuh individu lainnya
                new_guy.vel *= -1
                new_guy.killswitch(
                    self.waktu_pemulihan, self.imun_recovery
                )
                self.terinfeksi_container.add(new_guy)
                self.all_container.add(new_guy)

            # Individu yang berhasil sembuh (individu - recovery)
            recovered = []
            for guy in self.terinfeksi_container:
                if guy.recovered:
                    new_guy = guy.respawn(GREEN)
                    self.recovered_container.add(new_guy)
                    self.all_container.add(new_guy)
                    recovered.append(guy)

            if len(recovered) > 0:
                self.terinfeksi_container.remove(*recovered)
                self.all_container.remove(*recovered)

            self.all_container.draw(screen)

            del stats_graph
            stats.unlock()
            screen.blit(stats, stats_pos)
            pygame.display.flip()

            clock.tick(30)

        pygame.quit()


if __name__ == "__main__":
    # Ukuran ruang simulasi : 20x20 unit (20 CM -> 759,6 Pixel)
    virus = Simulation(756, 756)

    # Jumlah Individu
    virus.jumlah_individu = 200

    # Merupakan Individu yang tidak bergerak,
    # yaitu didapat dari 100% dikurang dengan
    # Probabilitas individu yang bergerak yaitu 80% maka didapat = 200 * 20%
    virus.probabilitas_Tidakbergerak = 40

    # Rasio individu terinfeksi
    virus.individu_terinfeksi = 10

    waktu_pulih = 10
    # Waktu pemulihan dalam timestamp, yaitu = 10
    virus.waktu_pemulihan = 100

    # Rate waktu imunitas untuk individu yang terjangkit agar dapat sembuh
    virus.imun_recovery = 0.2

    virus.start(randomize=True)

print("Waktu pemulihannya adalah : ", waktu_pulih, " - hari")
