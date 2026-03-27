# Úloha odvozená z rovnice (23) z článku — validační případ pro `p=2` a hlavní úloha pro `p=3`

## 1. Cíl dokumentu

Tento dokument je samostatné zadání k implementaci okrajové úlohy typu (23) ze zdrojového článku, specializované na

- oblast `Ω = (0,1)^2` (jednotkový čtverec),
- pravou nelinearitu `g(u) = arctan(u+1)`,
- zatížení `h(x) = 0`.

Dokument obsahuje dvě verze téže úlohy:

1. **validační úlohu pro `p=2`**, tj. semilineární Dirichletův problém,
2. **hlavní úlohu pro `p=3`**, tj. kvazilineární problém s `3`-Laplaciánem.

Validační případ `p=2` je určen k ověření:

- správnosti slabé formulace,
- sestavení diskrétního rezidua,
- implementace Dirichletovy podmínky,
- nelineárního členu `arctan(u+1)` a jeho derivace,
- základní konvergence diskrétního řešení.

Hlavní případ `p=3` je cílová, podstatně těžší úloha.

---

## 2. Obecný tvar úlohy (23)

V článku je uvažován problém

```math
\begin{cases}
-\Delta_p u = \lambda_1 |u|^{p-2}u + g(u) - h(x) & \text{v } \Omega,\\
u=0 & \text{na } \partial\Omega,
\end{cases}
```

kde

```math
\Delta_p u = \operatorname{div}\bigl(|\nabla u|^{p-2}\nabla u\bigr)
```

je `p`-Laplacián, `λ₁` je první vlastní číslo Dirichletova `p`-Laplaciánu a `φ₁` příslušná kladná vlastní funkce.

V tomto zadání fixujeme

```math
\Omega=(0,1)^2,
\qquad
h(x)=0,
\qquad
g(u)=\arctan(u+1).
```

Tedy obecně řešíme

```math
\begin{cases}
-\Delta_p u = \lambda_1 |u|^{p-2}u + \arctan(u+1) & \text{v } (0,1)^2,\\
u=0 & \text{na } \partial(0,1)^2.
\end{cases}
```

---

## 3. Pomocné funkce společné pro `p=2` i `p=3`

### 3.1 Nelinearita

```math
g(u)=\arctan(u+1).
```

Její derivace je

```math
g'(u)=\frac{1}{1+(u+1)^2}.
```

### 3.2 Primitivní funkce

Definujeme

```math
G(t)=\int_0^t g(s)\,ds=\int_0^t \arctan(s+1)\,ds.
```

Explicitně:

```math
G(t)
=
(t+1)\arctan(t+1)
-\frac12\ln\bigl(1+(t+1)^2\bigr)
-\left(\frac\pi4-\frac12\ln 2\right).
```

Tato volba konstanty zajišťuje `G(0)=0`.

### 3.3 Funkce `F` z věty o řešitelnosti

Ve větě 6 z článku vystupuje funkce

```math
F(x)=\frac{p}{x}\int_0^x g(s)\,ds-g(x)=\frac{p}{x}G(x)-\arctan(x+1).
```

Pro naši volbu `g(u)=arctan(u+1)` platí asymptoticky

```math
\lim_{x\to +\infty}F(x)=\frac{(p-1)\pi}{2},
\qquad
\lim_{x\to -\infty}F(x)=-\frac{(p-1)\pi}{2}.
```

Tedy speciálně:

- pro `p=2`:

```math
\lim_{x\to +\infty}F(x)=\frac\pi2,
\qquad
\lim_{x\to -\infty}F(x)=-\frac\pi2,
```

- pro `p=3`:

```math
\lim_{x\to +\infty}F(x)=\pi,
\qquad
\lim_{x\to -\infty}F(x)=-\pi.
```

---

## 4. Validační úloha pro `p=2`

## 4.1 Silná formulace

Validační úloha je

```math
\begin{cases}
-\Delta u = \lambda_1 u + \arctan(u+1) & \text{v } (0,1)^2,\\
u=0 & \text{na } \partial(0,1)^2.
\end{cases}
```

Na jednotkovém čtverci je první vlastní číslo Dirichletova Laplaceova operátoru explicitně známo:

```math
\lambda_1 = 2\pi^2.
```

Příslušná první vlastní funkce může být zvolena jako

```math
\varphi_1(x,y)=\sin(\pi x)\sin(\pi y).
```

Ta splňuje

```math
\begin{cases}
-\Delta \varphi_1 = 2\pi^2\varphi_1 & \text{v } (0,1)^2,\\
\varphi_1=0 & \text{na } \partial(0,1)^2,
\end{cases}
```

a navíc `φ₁>0` v interioru oblasti.

## 4.2 Slabá formulace

Hledá se

```math
u\in H_0^1(\Omega)
```

tak, aby pro každé `v\in H_0^1(\Omega)` platilo

```math
\int_\Omega \nabla u\cdot\nabla v\,dx
-
2\pi^2\int_\Omega uv\,dx
-
\int_\Omega \arctan(u+1)v\,dx
=0.
```

Ekvivalentně:

```math
\int_\Omega \nabla u\cdot\nabla v\,dx
=
2\pi^2\int_\Omega uv\,dx
+
\int_\Omega \arctan(u+1)v\,dx.
```

## 4.3 Variační formulace

Hledané řešení odpovídá kritickému bodu funkcionálu

```math
J_2(u)
=
\frac12\int_\Omega |\nabla u|^2\,dx
-
\frac{2\pi^2}{2}\int_\Omega u^2\,dx
-
\int_\Omega G(u)\,dx.
```

Tj.

```math
J_2(u)
=
\frac12\int_\Omega |\nabla u|^2\,dx
-
\pi^2\int_\Omega u^2\,dx
-
\int_\Omega G(u)\,dx.
```

## 4.4 Linearizace / Newtonův krok

Reziduum je

```math
R_2(u;v)
=
\int_\Omega \nabla u\cdot\nabla v\,dx
-
2\pi^2\int_\Omega uv\,dx
-
\int_\Omega \arctan(u+1)v\,dx.
```

Jeho Gâteauxova derivace ve směru `δu` je

```math
DR_2(u)[\delta u;v]
=
\int_\Omega \nabla\delta u\cdot\nabla v\,dx
-
2\pi^2\int_\Omega \delta u\,v\,dx
-
\int_\Omega \frac{1}{1+(u+1)^2}\,\delta u\,v\,dx.
```

## 4.5 Existence slabého řešení

Ze specializace věty 6 z článku plyne:

> **Věta (řešitelnost pro `p=2`).**
> Pro `Ω=(0,1)^2`, `h=0` a `g(u)=arctan(u+1)` má úloha
>
> ```math
> \begin{cases}
> -\Delta u = 2\pi^2 u + \arctan(u+1) & \text{v } (0,1)^2,\\
> u=0 & \text{na } \partial(0,1)^2
> \end{cases}
> ```
>
> alespoň jedno slabé řešení `u\in H_0^1((0,1)^2)`.

**Ověření předpokladů věty:**

1. `g(u)=arctan(u+1)` je omezená funkce, takže

```math
\lim_{|u|\to\infty}\frac{g(u)}{|u|^{p-1}}=0
```

je pro `p=2` splněno.

2. Protože `h=0`, prostřední člen v podmínce (25) je nulový.

3. Jelikož

```math
\lim_{x\to -\infty}F(x)=-\frac\pi2,
\qquad
\lim_{x\to +\infty}F(x)=\frac\pi2,
```

a `φ₁>0`, je

```math
-\frac\pi2\int_\Omega \varphi_1\,dx
< 0 <
\frac\pi2\int_\Omega \varphi_1\,dx,
```

takže podmínka (25) je splněna.

## 4.6 Známe přesné analytické řešení?

**Ne.** Pro zadanou úlohu

```math
-\Delta u = 2\pi^2 u + \arctan(u+1),
\qquad u|_{\partial\Omega}=0,
```

není známo jednoduché uzavřené analytické řešení ve smyslu explicitního vzorce. Pro validační účely je proto třeba použít:

- kontrolu slabého rezidua,
- kontrolu poklesu Newtonovy/Picardovy chyby,
- zjemňování sítě a porovnání diskrétních řešení,
- případně porovnání dvou nezávislých implementací.

Validační role případu `p=2` spočívá v tom, že:

- lineární část je zcela explicitní,
- `λ₁` i `φ₁` jsou známy přesně,
- tangentní operátor je snadno sestavitelný,
- problém je numericky výrazně méně citlivý než případ `p=3`.

---

## 5. Hlavní úloha pro `p=3`

Toto je cílová úloha k řešení.

## 5.1 Silná formulace

Hledá se funkce `u : Ω \to \mathbb{R}` taková, že

```math
\begin{cases}
-\Delta_3 u = \lambda_1 |u|u + \arctan(u+1) & \text{v } \Omega=(0,1)^2,\\
u=0 & \text{na } \partial\Omega,
\end{cases}
```

kde

```math
\Delta_3 u = \operatorname{div}\bigl(|\nabla u|\nabla u\bigr).
```

Tedy explicitně

```math
-\operatorname{div}\bigl(|\nabla u|\nabla u\bigr)
=
\lambda_1 |u|u + \arctan(u+1)
\qquad \text{v } (0,1)^2,
```

s homogenní Dirichletovou podmínkou.

## 5.2 První vlastní číslo `3`-Laplaciánu

Na rozdíl od případu `p=2` zde **není na jednotkovém čtverci k dispozici jednoduchý explicitní vzorec pro `λ₁` ani pro `φ₁`**. Proto musí být `λ₁` v rámci workflow získáno numericky z vlastního problému

```math
\begin{cases}
-\Delta_3 \varphi_1 = \lambda_1 |\varphi_1|\varphi_1 & \text{v } \Omega,\\
\varphi_1=0 & \text{na } \partial\Omega,
\end{cases}
```

s normalizací například

```math
\int_\Omega |\varphi_1|^3\,dx = 1.
```

Pak platí

```math
\lambda_1 = \int_\Omega |\nabla\varphi_1|^3\,dx.
```

Ekvivalentně lze `λ₁` definovat Rayleighovým podílem

```math
\lambda_1
=
\min\left\{
\int_\Omega |\nabla v|^3\,dx
\;\middle|\;
 v\in W_0^{1,3}(\Omega),\;
 \int_\Omega |v|^3\,dx=1
\right\}.
```

Pro implementaci je tedy třeba:

1. nejprve spočítat aproximaci `λ₁`,
2. poté tuto hodnotu použít v hlavní nelineární úloze.

## 5.3 Slabá formulace

Hledá se

```math
u\in W_0^{1,3}(\Omega)
```

tak, aby pro každé `v\in W_0^{1,3}(\Omega)` platilo

```math
\int_\Omega |\nabla u|\nabla u\cdot\nabla v\,dx
-
\lambda_1\int_\Omega |u|u\,v\,dx
-
\int_\Omega \arctan(u+1)v\,dx
=0.
```

## 5.4 Variační formulace

Úloha odpovídá kritickým bodům funkcionálu

```math
J_3(u)
=
\frac13\int_\Omega |\nabla u|^3\,dx
-
\frac{\lambda_1}{3}\int_\Omega |u|^3\,dx
-
\int_\Omega G(u)\,dx.
```

## 5.5 Derivace rezidua

Reziduum je

```math
R_3(u;v)
=
\int_\Omega |\nabla u|\nabla u\cdot\nabla v\,dx
-
\lambda_1\int_\Omega |u|u\,v\,dx
-
\int_\Omega \arctan(u+1)v\,dx.
```

Formální Gâteauxova derivace ve směru `δu` je

```math
DR_3(u)[\delta u;v]
=
\int_\Omega D\bigl(|\nabla u|\nabla u\bigr)[\nabla\delta u]\cdot\nabla v\,dx
-
\lambda_1\int_\Omega 2|u|\,\delta u\,v\,dx
-
\int_\Omega \frac{1}{1+(u+1)^2}\,\delta u\,v\,dx,
```
```
kde pro vektor `z\in\mathbb{R}^2`, `z\neq0`, platí

```math
D(|z|z)[\eta]
=
|z|\,\eta + \frac{z\cdot \eta}{|z|}\,z.
```

Prakticky tedy

```math
D\bigl(|\nabla u|\nabla u\bigr)[\nabla\delta u]
=
|\nabla u|\,\nabla\delta u
+
\frac{\nabla u\cdot\nabla\delta u}{|\nabla u|}\,\nabla u,
\qquad \text{pokud } \nabla u\neq 0.
```

### Poznámka k implementaci

Tato linearizace je citlivá v místech, kde `|∇u|` je velmi malé. Proto je vhodné v Newtonově typu řešiči použít regularizaci, například náhradou

```math
|\nabla u| \approx \sqrt{|\nabla u|^2+\varepsilon^2}
```

s malým `ε>0`, nebo místo čistého Newtona zvolit robustnější nelineární metodu (tlumený Newton, Picard, gradientní/pseudotime pokračování apod.).

## 5.6 Existence slabého řešení

Ze specializace věty 6 z článku plyne:

> **Věta (řešitelnost pro `p=3`).**
> Pro `Ω=(0,1)^2`, `h=0` a `g(u)=arctan(u+1)` má úloha
>
> ```math
> \begin{cases}
> -\Delta_3 u = \lambda_1 |u|u + \arctan(u+1) & \text{v } (0,1)^2,\\
> u=0 & \text{na } \partial(0,1)^2
> \end{cases}
> ```
>
> alespoň jedno slabé řešení `u\in W_0^{1,3}((0,1)^2)`.

**Ověření předpokladů věty:**

1. Pro `p=3` je třeba ověřit

```math
\lim_{|u|\to\infty}\frac{g(u)}{|u|^2}=0,
```

což platí, protože `arctan(u+1)` je omezená funkce.

2. Pro `h=0` je prostřední člen v (25) nulový.

3. Jelikož

```math
\lim_{x\to -\infty}F(x)=-\pi,
\qquad
\lim_{x\to +\infty}F(x)=\pi,
```

a první vlastní funkce `φ₁` je kladná, dostáváme

```math
-\pi\int_\Omega \varphi_1\,dx
< 0 <
\pi\int_\Omega \varphi_1\,dx,
```

takže podmínka (25) je splněna.

---

## 6. Doporučené pořadí implementace

1. **Validační případ `p=2`.**
   - použít explicitní `λ₁ = 2π²`,
   - implementovat slabou formulaci,
   - ověřit reziduum a konvergenci při zjemňování sítě.

2. **Samostatně spočítat `λ₁` pro `p=3`.**
   - řešit vlastní problém pro `3`-Laplacián,
   - znormalizovat `φ₁` pomocí `∫|φ₁|^3 = 1`,
   - získanou hodnotu `λ₁` uložit jako vstup do hlavní úlohy.

3. **Hlavní úloha `p=3`.**
   - řešit slabou formulaci s vypočteným `λ₁`,
   - použít robustní nelineární řešič,
   - sledovat normu rezidua a citlivost na počáteční aproximaci.

4. **Volitelně:** použít diskrétní řešení z `p=2` jako počáteční odhad pro `p=3`.

---

## 7. Stručné zadání k předání implementátorovi

### Validační úloha

Na oblasti `Ω=(0,1)^2` implementovat řešení problému

```math
\begin{cases}
-\Delta u = 2\pi^2 u + \arctan(u+1) & \text{v } \Omega,\\
u=0 & \text{na } \partial\Omega.
\end{cases}
```

Použít slabou formulaci

```math
\int_\Omega \nabla u\cdot\nabla v\,dx
-
2\pi^2\int_\Omega uv\,dx
-
\int_\Omega \arctan(u+1)v\,dx
=0
\qquad \forall v\in H_0^1(\Omega).
```

Přesné analytické řešení není k dispozici; validace má proběhnout pomocí rezidua a síťové konvergence.

### Hlavní úloha

Na stejné oblasti implementovat řešení problému

```math
\begin{cases}
-\operatorname{div}\bigl(|\nabla u|\nabla u\bigr)
=
\lambda_1 |u|u + \arctan(u+1) & \text{v } \Omega,\\
u=0 & \text{na } \partial\Omega,
\end{cases}
```

kde `λ₁` je první vlastní číslo Dirichletova `3`-Laplaciánu na `Ω=(0,1)^2`, získané numericky z vlastního problému

```math
\begin{cases}
-\Delta_3 \varphi_1 = \lambda_1 |\varphi_1|\varphi_1 & \text{v } \Omega,\\
\varphi_1=0 & \text{na } \partial\Omega,\\
\int_\Omega |\varphi_1|^3\,dx=1.
\end{cases}
```

Hlavní slabá formulace je

```math
\int_\Omega |\nabla u|\nabla u\cdot\nabla v\,dx
-
\lambda_1\int_\Omega |u|u\,v\,dx
-
\int_\Omega \arctan(u+1)v\,dx
=0
\qquad \forall v\in W_0^{1,3}(\Omega).
```

Ze specializované věty o řešitelnosti plyne existence alespoň jednoho slabého řešení jak pro validační případ `p=2`, tak pro hlavní případ `p=3`.
