---
name: Quiet Futurism
colors:
  surface: '#111318'
  surface-dim: '#111318'
  surface-bright: '#37393e'
  surface-container-lowest: '#0c0e12'
  surface-container-low: '#1a1c20'
  surface-container: '#1e2024'
  surface-container-high: '#282a2e'
  surface-container-highest: '#333539'
  on-surface: '#e2e2e8'
  on-surface-variant: '#bac9cb'
  inverse-surface: '#e2e2e8'
  inverse-on-surface: '#2f3035'
  outline: '#849395'
  outline-variant: '#3b494b'
  surface-tint: '#00dbe9'
  primary: '#88f5ff'
  on-primary: '#00363a'
  primary-container: '#00ddeb'
  on-primary-container: '#005d63'
  inverse-primary: '#006970'
  secondary: '#aec6ff'
  on-secondary: '#002e6a'
  secondary-container: '#0361d1'
  on-secondary-container: '#dbe4ff'
  tertiary: '#ffdfb0'
  on-tertiary: '#422c00'
  tertiary-container: '#ffbc3b'
  on-tertiary-container: '#6f4d00'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#7df4ff'
  primary-fixed-dim: '#00dbe9'
  on-primary-fixed: '#002022'
  on-primary-fixed-variant: '#004f54'
  secondary-fixed: '#d8e2ff'
  secondary-fixed-dim: '#aec6ff'
  on-secondary-fixed: '#001a42'
  on-secondary-fixed-variant: '#004396'
  tertiary-fixed: '#ffdeab'
  tertiary-fixed-dim: '#fdbb3a'
  on-tertiary-fixed: '#271900'
  on-tertiary-fixed-variant: '#5f4100'
  background: '#111318'
  on-background: '#e2e2e8'
  surface-variant: '#333539'
  surface-deep: '#0D1015'
  surface-default: '#11151C'
  surface-elevated: '#151A22'
  surface-high: '#1A2029'
  research-violet: '#9B5CFF'
  status-benign: '#00D99A'
  status-warning: '#FFB800'
  status-critical: '#FF4354'
typography:
  headline-lg:
    fontFamily: Geist
    fontSize: 48px
    fontWeight: '600'
    lineHeight: '1.1'
    letterSpacing: -0.02em
  headline-lg-mobile:
    fontFamily: Geist
    fontSize: 32px
    fontWeight: '600'
    lineHeight: '1.2'
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Geist
    fontSize: 24px
    fontWeight: '500'
    lineHeight: '1.3'
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: '1.6'
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.5'
  body-sm:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: '1.5'
  technical-label:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: '500'
    lineHeight: '1.2'
    letterSpacing: 0.05em
  technical-data:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '400'
    lineHeight: '1.0'
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  unit: 4px
  gutter: 24px
  margin-mobile: 16px
  margin-desktop: 32px
  panel-gap: 16px
  max-width: 1440px
---

## Brand & Style

The design system is defined by the philosophy of **Quiet Futurism**. It rejects the aggressive, high-contrast aesthetics often associated with artificial intelligence in favor of a restrained, high-fidelity scientific environment. The visual language evokes a premium clinical workstation—calm, authoritative, and surgically precise.

The style blends **Minimalism** with **Glassmorphism**, using subtle translucency and tonal layering to suggest depth without visual clutter. It prioritizes information density and technical clarity, utilizing cool-gray borders and a "Deep Space" canvas to create a focused atmosphere for high-stakes research and diagnostic work. Every element feels engineered rather than merely designed.

## Colors

The color system is optimized for long-duration laboratory use, minimizing eye strain through a dark-mode-first architecture.

- **Foundational Neutrals:** The background (#090B0F) serves as the primary canvas. Hierarchy is built through progressive "Surface" tiers, moving from Deep to High as elements move closer to the user in the Z-axis.
- **Electric Cyan (AI/Primary):** This high-energy hue is reserved for active intelligence states, primary actions, and system-driven insights.
- **Secondary Blue:** Used for navigation and interactive architecture that isn't directly AI-driven.
- **Semantic Palette:** Research Violet is dedicated to architecture and explainability visualizations. Emerald, Amber, and Red are used strictly for clinical diagnostics (Benign, Warning, Critical).
- **Cool-Gray Borders:** Use `rgba(255, 255, 255, 0.08)` for structural containment, creating "ghost" outlines that define edges without adding visual weight.

## Typography

Typography is the primary vehicle for scientific precision. 

- **Geist:** Used for headers and structural wayfinding. Its geometric, slightly condensed nature feels architectural and modern.
- **Inter:** Used for clinical notes, descriptions, and standard UI labels. It provides a necessary humanist balance to the technical environment.
- **JetBrains Mono:** This is the "Lab Grade" font. It must be used for all raw data, model coordinates, confidence scores, and metadata. In technical labels, use uppercase with increased letter spacing to signal its status as system-generated telemetry.

## Layout & Spacing

The layout is built on a **12-column fixed grid** with a maximum width of 1440px, transitioning to a fluid model for the internal contents of individual panels.

- **Modular Workstations:** Viewports are composed of "Modules"—independent, often resizable panels that house specific tools or data views. 
- **The 4px Rhythm:** All spacing, padding, and alignment must adhere to a 4px base unit. 
- **Mobile Reflow:** On small screens, the 12-column grid collapses into a single vertical stack. Data-heavy tables should utilize horizontal overflow with the primary key column (e.g., Patient ID) pinned to the left.

## Elevation & Depth

This system eschews traditional drop shadows in favor of **Tonal Layering** and **Backdrop Blurs**.

- **Surface Tiers:** Depth is communicated by color shifting. The base canvas is the darkest, with progressively lighter surfaces representing "higher" interactive layers.
- **Optical Boundaries:** Use 1px cool-gray borders at low opacity (8-12%) to define surface edges. For "Elevated" or "High" surfaces (modals/popovers), apply a 16px - 24px backdrop blur to create a frosted glass effect that maintains the context of the underlying diagnostic data.
- **Active Glow:** Interactive elements in an "active" or "focused" state may use a subtle, 0-spread outer glow using the primary Cyan color to indicate system focus.

## Shapes

The shape language is **Technical and Precise**. The standard radius is 4px (`rounded-sm`), applied to buttons, input fields, and small UI components to suggest engineering precision. 

Larger containers and cards utilize an 8px (`rounded-lg`) radius to provide structural stability. Avoid pill-shaped or highly rounded elements, as they undermine the "Scientific Machine" aesthetic. Geometric precision—sharp but not harsh—is the goal.

## Components

- **Buttons:** Primary buttons use a solid Electric Cyan fill with black text. Secondary buttons are transparent with a 1px Cyan border at 30% opacity. Ghost buttons are for utility actions and should use `technical-label` typography.
- **Input Fields:** These should be slightly darker than the surface they reside on. Upon focus, the border transitions to Primary Cyan with a subtle 2px outer glow.
- **Cards:** Utilize the `Surface` tiering. Card headers should be separated from content by a 1px low-opacity horizontal rule.
- **Technical Chips:** Rectangular indicators with a 2px radius. Use monochrome backgrounds for labels and semantic colors (Emerald/Red) only for status-critical data.
- **The Scanner Marquee:** A system-specific component for dermoscopic images. Use a 1px Primary Cyan dashed border with a high-frequency "scanning" pulse animation.
- **Status Dots:** Small circles with a 4px blur box-shadow in their respective semantic color (Emerald, Amber, Red) to give the dashboard a "live" system feel.
- **Data Tables:** Use a zebra-stripe pattern with `Deep Surface` and `Surface` colors. Typography should be `technical-data` (JetBrains Mono) for all numerical entries.