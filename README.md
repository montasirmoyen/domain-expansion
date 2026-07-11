# domain-expansion

> [!NOTE]  
> This project may contain minor spoilers to the Jujutsu Kaisen series 😅

This program uses a webcam to detect specific hand gestures and identifies which Domain Expansion hand sign they correspond to. When a gesture is recognized, the program displays the name of the Domain Expansion associated with that hand sign.

Web version: https://domain-expansion-jjk.vercel.app/

<details>
<summary>Current Domain Expansions</summary>
<ul>
<li><b>Satoru Gojo's "Unlimited Void"</b> (無量空処, むりょうくうしょ, Muryōkūsho)</li>
<li><b>Ryomen Sukuna's "Malevolent Shrine"</b> (伏魔御廚子, ふくまみづし, Fukuma Mizushi)</li>
<li><b>Mahito's "Self-Embodiment of Perfection"</b> (自閉円頓裹, じへいえんどんか, Jihei Endonka)</li>
<li><b>Yuji Itadori's Unnamed Domain Expansion</b> (※名称不明, めいしょうふめい, Meishō Fumei)</li>
<li><b>Kinji Hakari's "Idle Death Gamble"</b> (坐殺博徒, ざさつばくと, Zasatsu Bakuto)</li>
<li><b>Yuta Okkotsu's "Authentic Mutual Love"</b> (真贋相愛, しんがんそうあい, Shingan Sōai)</li>
</ul>
</details>

### Backstory

![Ryomen Sukuna's "Malevolent Shrine"](/public/ms-de.png)

A "Domain Expansion" (領域展開, りょういきてんかい, Ryōiki Tenkai) is an advanced barrier technique and is considered the pinnacle of jujutsu sorcery in the popular Japanese dark fantasy manga and anime series Jujutsu Kaisen, created by Gege Akutami [(Wiki)](https://jujutsu-kaisen.fandom.com/wiki/Domain_Expansion).

[How to perform each Domain Expansion?](https://www.wikihow.com/Domain-Expansion-Hand-Sign)

# Usage

This repository contains two implementations of the project:

- **Python version (`python-proj/`)** - the original OpenCV and MediaPipe desktop application.
- **Web version (`dex-client/`)** — a browser-based React, MediaPipe, and Three.js experience with webcam gesture detection and interactive Domain Expansion visualizations.

### Python Version

```bash
git clone https://github.com/montasirmoyen/domain-expansion
cd domain-expansion

pip install -r requirements.txt

python3 src/main.py
````

Pressing **Q** on your keyboard closes the program.

### Web Version

Make sure [Node.js](https://nodejs.org/) and npm are installed, then run:

```bash
cd dex-client
npm i
npm run dev
```

Open the local URL printed by Vite (usually `http://localhost:5173`) and allow camera access when prompted. Webcam processing happens locally in the browser.

## Model Attribution

[Malevolent Shrine – Jujutsu Kaisen](https://sketchfab.com/3d-models/malevolent-shrine-jujutsu-kaisen-b0e9e4a8682c43acaf9051c507c2b13b) by [TSK_VITANK](https://sketchfab.com/TSK_VITANK) is licensed under [Creative Commons Attribution-NonCommercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/). This model may not be used commercially.
