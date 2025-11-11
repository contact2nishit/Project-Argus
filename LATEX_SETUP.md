# LaTeX Setup Guide for Project Argus Research Paper

## 📚 Overview
This guide will help you set up LaTeX on your system and compile the Project Argus research paper.

---

## 🔧 Installation

### Ubuntu/Debian (Recommended for this project)
```bash
# Update package list
sudo apt update

# Install LaTeX and required packages (this takes ~2GB disk space)
sudo apt install -y \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-bibtex-extra \
    biber

# Optional: Install additional tools
sudo apt install -y \
    texlive-science \
    texlive-publishers \
    latexmk
```

### macOS
```bash
# Using Homebrew
brew install --cask mactex

# Or install BasicTeX (smaller)
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install collection-latexextra
```

### Windows
1. Download and install **MiKTeX**: https://miktex.org/download
2. Or install **TeX Live**: https://tug.org/texlive/windows.html
3. Recommended: Install **TeXstudio** as an editor: https://www.texstudio.org/

---

## 📁 Project Structure

```
Project-Argus/
├── paper/
│   ├── main.tex              # Main paper file
│   ├── references.bib        # Bibliography (to be created)
│   ├── figures/              # Store your figures here
│   │   └── ppo_training_results.png
└── LATEX_SETUP.md           # This file
```

---

## ✍️ Compiling the Paper

### Method 1: Using pdflatex (Recommended)
```bash
# Navigate to the paper directory
cd /home/nishit/Project-Argus/paper

# Compile the paper (run 2-3 times for references)
pdflatex main.tex
pdflatex main.tex

# If using BibTeX for references
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Method 2: Using latexmk (Auto-compilation)
```bash
cd /home/nishit/Project-Argus/paper

# Compile with automatic dependency handling
latexmk -pdf main.tex

# Enable continuous preview (recompiles on save)
latexmk -pdf -pvc main.tex
```

### Method 3: One-Line Compilation
```bash
cd /home/nishit/Project-Argus/paper && pdflatex main.tex && pdflatex main.tex
```

---

## 🖼️ Adding Figures

### Step 1: Create figures directory
```bash
mkdir -p /home/nishit/Project-Argus/paper/figures
```

### Step 2: Copy your training plots
```bash
# Copy PPO training results
cp /home/nishit/Project-Argus/ppo_training_results.png \
   /home/nishit/Project-Argus/paper/figures/

# Copy any other figures
cp /path/to/your/figure.png paper/figures/
```

### Step 3: Include in LaTeX
Uncomment and modify in `main.tex`:
```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{figures/ppo_training_results.png}
    \caption{PPO training performance over 1000 episodes.}
    \label{fig:training}
\end{figure}
```

---

## 📖 Using BibTeX for References

### Step 1: Create bibliography file
Create `paper/references.bib`:
```bibtex
@article{author2023rescue,
  title={Title of the Paper},
  author={Author, First and Author, Second},
  journal={Journal Name},
  volume={10},
  number={2},
  pages={123--145},
  year={2023},
  publisher={Publisher Name}
}

@inproceedings{author2024rl,
  title={Reinforcement Learning for Robotics},
  author={Author, Name},
  booktitle={Conference Name},
  pages={1--10},
  year={2024}
}
```

### Step 2: Cite in your paper
```latex
Recent work in rescue robotics \cite{author2023rescue} has shown...
```

### Step 3: Compile with BibTeX
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## 🎨 Recommended VS Code Extensions

If using VS Code (already installed):
```bash
# Install LaTeX Workshop extension
code --install-extension James-Yu.latex-workshop
```

### VS Code Configuration
Add to `.vscode/settings.json`:
```json
{
    "latex-workshop.latex.autoBuild.run": "onSave",
    "latex-workshop.view.pdf.viewer": "tab",
    "latex-workshop.latex.recipes": [
        {
            "name": "pdflatex → bibtex → pdflatex × 2",
            "tools": [
                "pdflatex",
                "bibtex",
                "pdflatex",
                "pdflatex"
            ]
        }
    ]
}
```

---

## 🚀 Quick Start Workflow

### For Team Members (First Time)
```bash
# 1. Install LaTeX
sudo apt update && sudo apt install -y texlive-latex-base texlive-latex-extra \
    texlive-fonts-recommended texlive-fonts-extra texlive-bibtex-extra biber

# 2. Navigate to paper directory
cd /home/nishit/Project-Argus/paper

# 3. Compile the paper
pdflatex main.tex
pdflatex main.tex

# 4. View the PDF
xdg-open main.pdf  # Linux
# or: open main.pdf  # macOS
# or: start main.pdf  # Windows
```

### Daily Writing Workflow
```bash
# 1. Edit main.tex with your favorite editor
nano main.tex   # or vim, code, etc.

# 2. Compile (while in paper/ directory)
pdflatex main.tex

# 3. View changes
xdg-open main.pdf

# 4. Repeat steps 1-3 as needed
```

---

## 🔍 Common LaTeX Commands

### Document Structure
```latex
\section{Section Title}
\subsection{Subsection Title}
\subsubsection{Subsubsection Title}
```

### Text Formatting
```latex
\textbf{bold text}
\textit{italic text}
\texttt{monospace/code}
\underline{underlined}
```

### Lists
```latex
% Bulleted list
\begin{itemize}
    \item First item
    \item Second item
\end{itemize}

% Numbered list
\begin{enumerate}
    \item First item
    \item Second item
\end{enumerate}
```

### Math Equations
```latex
% Inline math
The equation $y = mx + b$ represents...

% Display math
\begin{equation}
    E = mc^2
    \label{eq:einstein}
\end{equation}

% Reference equation
As shown in Equation \ref{eq:einstein}...
```

### Tables
```latex
\begin{table}[h]
\centering
\caption{Your table caption}
\label{tab:mytable}
\begin{tabular}{lcc}
\toprule
\textbf{Column 1} & \textbf{Column 2} & \textbf{Column 3} \\
\midrule
Row 1 & Data & Data \\
Row 2 & Data & Data \\
\bottomrule
\end{tabular}
\end{table}
```

### Figures
```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=0.8\columnwidth]{figures/myimage.png}
    \caption{Caption describing the figure}
    \label{fig:myfig}
\end{figure}

% Reference figure
See Figure \ref{fig:myfig} for details...
```

### Citations
```latex
Recent work \cite{author2023} shows...
Multiple citations \cite{author2023,author2024}.
```

---

## 🐛 Troubleshooting

### Error: "pdflatex: command not found"
```bash
# Install LaTeX
sudo apt install texlive-latex-base
```

### Error: "File 'XXX.sty' not found"
```bash
# Install additional packages
sudo apt install texlive-latex-extra

# Or specific package
sudo apt install texlive-science  # for scientific packages
```

### Error: References not showing up
```bash
# You need to compile multiple times
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Error: Undefined citations
Make sure you're citing correctly and have run BibTeX:
```bash
bibtex main
pdflatex main.tex
```

### PDF not updating
```bash
# Clean auxiliary files
rm *.aux *.log *.bbl *.blg *.out

# Recompile from scratch
pdflatex main.tex
```

---

## 📊 Adding Your Results

### Update the Results Section
1. Run your experiments and save figures:
   ```bash
   python3 src/training/train_ppo.py
   ```

2. Copy figures to paper directory:
   ```bash
   cp ppo_training_results.png paper/figures/
   ```

3. Update table values in `main.tex` with your actual results

4. Uncomment figure code and add your plots

---

## 👥 Collaboration Tips

### Using Git with LaTeX
```bash
# Ignore auxiliary files
echo "*.aux
*.log
*.bbl
*.blg
*.out
*.toc
*.fdb_latexmk
*.fls
*.synctex.gz" >> .gitignore

# Only commit source files
git add paper/main.tex paper/references.bib paper/figures/
git commit -m "Update paper draft"
```

### Splitting Work
- Different teammates can work on different `\section{}`
- Use `\input{sections/introduction.tex}` to include separate files
- Regularly compile the full document to check formatting

### Version Control for Papers
```bash
# Tag major versions
git tag v1.0-draft
git tag v1.1-review
git tag v2.0-final

# Create branches for major revisions
git checkout -b revision-round-1
```

---

## 📝 Template Customization

### Change Conference Format
```latex
% IEEE Conference (current)
\documentclass[conference]{IEEEtran}

% IEEE Journal
\documentclass[journal]{IEEEtran}

% ACM Conference
\documentclass[sigconf]{acmart}

% AAAI Conference
\documentclass{aaai}
```

### Add More Packages
```latex
\usepackage{algorithm}     % For algorithms
\usepackage{listings}      % For code listings
\usepackage{subcaption}    % For subfigures
\usepackage{tikz}          % For diagrams
```

---

## ✅ Checklist Before Submission

- [ ] All figures included and referenced
- [ ] All tables filled with real data
- [ ] All citations added to references.bib
- [ ] Compiled at least 3 times for references
- [ ] No compilation errors or warnings
- [ ] Abstract updated with final results
- [ ] Author names and affiliations correct
- [ ] Page limit met (check conference requirements)
- [ ] PDF rendered correctly (check fonts, figures)
- [ ] Spell check completed
- [ ] All TODOs removed

---

## 📚 Useful Resources

- **Overleaf** (online LaTeX editor): https://www.overleaf.com
- **LaTeX Wikibook**: https://en.wikibooks.org/wiki/LaTeX
- **Detexify** (find symbol by drawing): http://detexify.kirelabs.org
- **IEEE Author Tools**: https://ieeeauthorcenter.ieee.org
- **LaTeX Table Generator**: https://www.tablesgenerator.com

---

## 🎯 Quick Reference Card

| Task | Command |
|------|---------|
| Compile paper | `pdflatex main.tex` |
| Compile with bibliography | `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex` |
| View PDF | `xdg-open main.pdf` |
| Clean auxiliary files | `rm *.aux *.log *.out` |
| Word count | `texcount main.tex` |
| Spell check | `aspell -c main.tex` |

---

## 💡 Pro Tips

1. **Compile often**: Catch errors early by compiling frequently
2. **Use comments**: Add `% TODO: ...` for things to fix later
3. **Version control**: Commit after each major section
4. **Backup**: Keep copies of your .tex files
5. **Read the log**: Error messages in `main.log` are helpful
6. **Use labels**: Always `\label{}` your sections, figures, equations
7. **Consistent naming**: Use descriptive names for labels (e.g., `fig:training-results`)

---

## 🆘 Getting Help

If you encounter issues:
1. Check the error message in the terminal
2. Look at `main.log` for detailed errors
3. Search for the error on https://tex.stackexchange.com
4. Ask your teammates or supervisor

---

**Happy Writing! 📝✨**

Generated for Project Argus team - November 2025
