# Apprendre l'IA — vidéo de formation (stage)

Vidéo pédagogique de ~90 secondes pour démarrer avec l'intelligence artificielle, pensée comme une formation d'accueil (onboarding) pour un stagiaire.

## Regarder la vidéo

- **[Apprendre l'IA (MP4, ~33 Mo)](apprendre_lia.mp4)** — 1280×720, 24 i/s, 84 s, avec bande-son d'ambiance
- Miniature : [poster.jpg](poster.jpg)

## Contenu de la vidéo

| Chapitre | Sujet |
|---|---|
| Intro | APPRENDRE L'IA |
| 01 · Définition | Qu'est-ce que l'IA ? |
| 02 · Mécanique | Comment une machine apprend |
| 03 · Paysage | Les grandes familles d'IA |
| 04 · Vocabulaire | Le vocabulaire à connaître |
| 05 · Outils | Ta boîte à outils de stagiaire |
| 06 · Parcours | Ton plan sur 4 semaines |
| 07 · Bonnes pratiques | Réussir ton stage avec l'IA |
| Outro | À toi de jouer ! |

## Régénérer la vidéo

Le script génère entièrement la vidéo par programmation (fonds en dégradé, réseau de neurones animé, transitions, barre de progression) — aucune ressource externe.

Prérequis :

```bash
pip install pillow numpy opencv-python matplotlib imageio-ffmpeg
```

Génération (la bande-son `soundtrack.py` est synthétisée puis incrustée automatiquement) :

```bash
python3 make_video.py apprendre_lia.mp4
```

(~1 minute de rendu ; le poster `poster.jpg` est produit au passage. Sans `imageio-ffmpeg`, la vidéo est rendue sans son.)

## Personnaliser

- les textes des chapitres : liste `chapters` dans `make_video.py` (titres, puces, durée de chaque séquence) ;
- les couleurs : accents définis en haut du script (cyan, violet, vert, ambre…) ;
- la musique : progression d'accords et tempo dans `soundtrack.py` ;
- le style : fond, polices, vitesse des animations.

Idem pour l'intro (`IntroSlide`) et la conclusion (`OutroSlide`).
