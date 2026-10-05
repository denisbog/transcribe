//! Palette, fonts and widget styles — the whole look of the app lives here.

use std::borrow::Cow;

use iced::border::{self, Border};
use iced::font::{Font, Weight};
use iced::widget::{
    button, container, progress_bar, scrollable, slider, text, toggler,
};
use iced::{Color, Shadow, Theme, Vector};

// ---------------------------------------------------------------- palette

pub const BG: Color = Color::from_rgb8(0x0a, 0x0d, 0x14);
pub const SURFACE: Color = Color::from_rgb8(0x13, 0x18, 0x21);
pub const SURFACE_HI: Color = Color::from_rgb8(0x1c, 0x24, 0x30);
pub const BORDER: Color = Color::from_rgb8(0x23, 0x2b, 0x38);
pub const TEXT: Color = Color::from_rgb8(0xe9, 0xee, 0xf5);
pub const TEXT_DIM: Color = Color::from_rgb8(0x9d, 0xa9, 0xba);
pub const TEXT_MUTED: Color = Color::from_rgb8(0x64, 0x70, 0x82);
pub const ACCENT: Color = Color::from_rgb8(0x6e, 0x8b, 0xff);
pub const ACCENT_HI: Color = Color::from_rgb8(0x8f, 0xa6, 0xff);
pub const ACCENT_LO: Color = Color::from_rgb8(0x53, 0x6f, 0xe8);
pub const WHITE: Color = Color::from_rgb8(0xf7, 0xf9, 0xff);

/// The accent at reduced opacity — used for tints and glows.
pub const fn accent_tint(alpha: f32) -> Color {
    alpha_color(ACCENT, alpha)
}

pub const fn alpha_color(color: Color, alpha: f32) -> Color {
    Color { a: alpha, ..color }
}

// ---------------------------------------------------------------- fonts

/// Fonts used across the UI. `display` for headings and numbers, `body` for
/// running text, `mono` for timestamps.
#[derive(Debug, Clone, Copy)]
pub struct Fonts {
    pub display: Font,
    pub body: Font,
    pub mono: Font,
}

impl Fonts {
    fn fallback() -> Self {
        Self {
            display: Font::DEFAULT,
            body: Font::DEFAULT,
            mono: Font::MONOSPACE,
        }
    }
}

/// Reads a handful of system fonts and returns them together with their bytes
/// (the bytes must be handed to iced, which does the real registration).
/// Missing files are silently ignored and fall back to iced's bundled font.
pub fn load_fonts() -> (Fonts, Vec<Cow<'static, [u8]>>) {
    const FILES: &[&str] = &[
        "/usr/share/fonts/truetype/Poppins-Medium.ttf",
        "/usr/share/fonts/truetype/Poppins-SemiBold.ttf",
        "/usr/share/fonts/truetype/Roboto-Regular.ttf",
        "/usr/share/fonts/truetype/Roboto-Medium.ttf",
        "/usr/share/fonts/truetype/SourceCodePro-Medium.otf",
        "/usr/share/fonts/truetype/SourceCodePro-Regular.otf",
    ];

    let mut bytes = Vec::new();
    let mut have = |path: &str| {
        let Ok(data) = std::fs::read(path) else {
            return false;
        };
        bytes.push(Cow::Owned(data));
        true
    };

    for file in FILES {
        have(file);
    }

    if bytes.is_empty() {
        return (Fonts::fallback(), bytes);
    }

    let fonts = Fonts {
        display: Font {
            weight: Weight::Medium,
            ..Font::with_name("Poppins")
        },
        body: Font::with_name("Roboto"),
        mono: Font {
            weight: Weight::Medium,
            ..Font::with_name("Source Code Pro")
        },
    };

    (fonts, bytes)
}

// ---------------------------------------------------------------- containers

/// The application background.
pub fn root(_theme: &Theme) -> container::Style {
    container::Style {
        background: Some(BG.into()),
        ..Default::default()
    }
}

/// A raised card.
pub fn card(_theme: &Theme) -> container::Style {
    container::Style {
        background: Some(SURFACE.into()),
        border: Border {
            color: BORDER,
            width: 1.0,
            radius: 16.0.into(),
        },
        shadow: Shadow {
            color: Color::from_rgba8(0, 0, 0, 0.45),
            offset: Vector::new(0.0, 10.0),
            blur_radius: 26.0,
        },
        ..Default::default()
    }
}

/// The "now playing" card: same as [`card`] but with a soft accent glow.
pub fn highlight_card(_theme: &Theme) -> container::Style {
    container::Style {
        background: Some(SURFACE.into()),
        border: Border {
            color: accent_tint(0.30),
            width: 1.0,
            radius: 16.0.into(),
        },
        shadow: Shadow {
            color: accent_tint(0.13),
            offset: Vector::new(0.0, 2.0),
            blur_radius: 28.0,
        },
        ..Default::default()
    }
}

/// A small pill for meta information (language, counts).
pub fn pill(_theme: &Theme) -> container::Style {
    container::Style {
        background: Some(SURFACE_HI.into()),
        border: border::rounded(999),
        ..Default::default()
    }
}

/// The 3 px accent bar marking the active sentence.
pub fn accent_bar(_theme: &Theme) -> container::Style {
    container::Style {
        background: Some(ACCENT.into()),
        border: border::rounded(2),
        ..Default::default()
    }
}

// ---------------------------------------------------------------- buttons

/// The primary transport button (filled, accent, rounded square).
pub fn play(_theme: &Theme, status: button::Status) -> button::Style {
    let background = match status {
        button::Status::Hovered => ACCENT_HI,
        button::Status::Pressed => ACCENT_LO,
        _ => ACCENT,
    };

    button::Style {
        background: Some(background.into()),
        text_color: WHITE,
        border: border::rounded(14),
        shadow: Shadow {
            color: accent_tint(0.35),
            offset: Vector::new(0.0, 6.0),
            blur_radius: 18.0,
        },
        ..Default::default()
    }
}

/// A quiet button: invisible until hovered.
pub fn ghost(_theme: &Theme, status: button::Status) -> button::Style {
    let background = match status {
        button::Status::Hovered => Some(SURFACE_HI.into()),
        button::Status::Pressed => Some(alpha_color(SURFACE_HI, 0.7).into()),
        _ => None,
    };

    button::Style {
        background,
        text_color: TEXT_DIM,
        border: border::rounded(10),
        ..Default::default()
    }
}

/// One word in the "now playing" panel.
pub fn word(_theme: &Theme, status: button::Status, active: bool) -> button::Style {
    let background = if active {
        Some(ACCENT.into())
    } else {
        match status {
            button::Status::Hovered => Some(SURFACE_HI.into()),
            _ => None,
        }
    };

    button::Style {
        background,
        text_color: if active { WHITE } else { TEXT },
        border: border::rounded(8),
        ..Default::default()
    }
}

/// A word in the "now playing" panel that belongs to a vocabulary pair.
pub fn word_pair(_theme: &Theme, status: button::Status, active: bool) -> button::Style {
    let background = if active {
        Some(ACCENT.into())
    } else {
        match status {
            button::Status::Hovered => Some(accent_tint(0.32).into()),
            _ => Some(accent_tint(0.18).into()),
        }
    };

    button::Style {
        background,
        text_color: if active { WHITE } else { ACCENT_HI },
        border: border::rounded(8),
        ..Default::default()
    }
}

/// A translation word highlighted because it is part of a vocabulary pair.
pub fn pair_chip(_theme: &Theme) -> container::Style {
    container::Style {
        background: Some(accent_tint(0.20).into()),
        border: border::rounded(5),
        ..Default::default()
    }
}

/// One sentence in the transcript list.
pub fn sentence(
    _theme: &Theme,
    status: button::Status,
    active: bool,
) -> button::Style {
    let background = if active {
        Some(accent_tint(0.14).into())
    } else {
        match status {
            button::Status::Hovered => Some(SURFACE_HI.into()),
            _ => None,
        }
    };

    button::Style {
        background,
        text_color: TEXT,
        border: border::rounded(10),
        ..Default::default()
    }
}

// ---------------------------------------------------------------- controls

/// The seek bar: slim rail, round handle that grows while hovering.
pub fn seek(_theme: &Theme, status: slider::Status) -> slider::Style {
    let radius = match status {
        slider::Status::Hovered | slider::Status::Dragged => 8.0,
        slider::Status::Active => 6.0,
    };

    slider::Style {
        rail: slider::Rail {
            backgrounds: (ACCENT.into(), SURFACE_HI.into()),
            width: 5.0,
            border: border::rounded(999),
        },
        handle: slider::Handle {
            shape: slider::HandleShape::Circle { radius },
            background: WHITE.into(),
            border_width: 0.0,
            border_color: Color::TRANSPARENT,
        },
    }
}

/// The volume bar: even slimmer, smaller handle.
pub fn volume(_theme: &Theme, status: slider::Status) -> slider::Style {
    let radius = match status {
        slider::Status::Hovered | slider::Status::Dragged => 6.0,
        slider::Status::Active => 5.0,
    };

    slider::Style {
        rail: slider::Rail {
            backgrounds: (TEXT_DIM.into(), SURFACE_HI.into()),
            width: 4.0,
            border: border::rounded(999),
        },
        handle: slider::Handle {
            shape: slider::HandleShape::Circle { radius },
            background: TEXT.into(),
            border_width: 0.0,
            border_color: Color::TRANSPARENT,
        },
    }
}

/// Thin, unobtrusive scrollbar that only shows a soft thumb.
pub fn scroll(_theme: &Theme, _status: scrollable::Status) -> scrollable::Style {
    let rail = scrollable::Rail {
        background: None,
        border: border::rounded(999),
        scroller: scrollable::Scroller {
            background: alpha_color(TEXT_MUTED, 0.7).into(),
            border: border::rounded(999),
        },
    };

    scrollable::Style {
        container: container::Style::default(),
        vertical_rail: rail,
        horizontal_rail: rail,
        gap: None,
        auto_scroll: scrollable::AutoScroll {
            background: SURFACE.into(),
            border: border::rounded(999),
            shadow: Shadow::default(),
            icon: TEXT,
        },
    }
}

/// The "Follow" switch.
pub fn toggle(_theme: &Theme, status: toggler::Status) -> toggler::Style {
    let is_toggled = match status {
        toggler::Status::Active { is_toggled }
        | toggler::Status::Hovered { is_toggled }
        | toggler::Status::Disabled { is_toggled } => is_toggled,
    };
    let hovered = matches!(status, toggler::Status::Hovered { .. });

    toggler::Style {
        background: if is_toggled {
            (if hovered { ACCENT_HI } else { ACCENT }).into()
        } else {
            SURFACE_HI.into()
        },
        background_border_width: 0.0,
        background_border_color: Color::TRANSPARENT,
        foreground: WHITE.into(),
        foreground_border_width: 0.0,
        foreground_border_color: Color::TRANSPARENT,
        text_color: Some(TEXT_DIM),
        border_radius: Some(999.0.into()),
        padding_ratio: 0.16,
    }
}

// ---------------------------------------------------------------- inputs


/// A small labelled chip that looks like a button (`±5 s`, `New article`).
pub fn chip(_theme: &Theme, status: button::Status) -> button::Style {
    let background = match status {
        button::Status::Hovered => SURFACE_HI,
        button::Status::Pressed => alpha_color(SURFACE_HI, 0.7),
        _ => SURFACE,
    };

    button::Style {
        background: Some(background.into()),
        text_color: TEXT,
        border: Border {
            color: BORDER,
            width: 1.0,
            radius: 10.0.into(),
        },
        ..Default::default()
    }
}

/// The one important action on a screen.
pub fn primary(_theme: &Theme, status: button::Status) -> button::Style {
    let background = match status {
        button::Status::Hovered => ACCENT_HI,
        button::Status::Pressed => ACCENT_LO,
        button::Status::Disabled => alpha_color(ACCENT, 0.35),
        _ => ACCENT,
    };

    button::Style {
        background: Some(background.into()),
        text_color: WHITE,
        border: border::rounded(10),
        shadow: Shadow {
            color: accent_tint(0.3),
            offset: Vector::new(0.0, 4.0),
            blur_radius: 14.0,
        },
        ..Default::default()
    }
}

/// A navigation tab: filled when selected, otherwise quiet.
pub fn tab(_theme: &Theme, status: button::Status, selected: bool) -> button::Style {
    let background = if selected {
        Some(alpha_color(ACCENT, 0.16).into())
    } else {
        match status {
            button::Status::Hovered => Some(SURFACE_HI.into()),
            _ => None,
        }
    };

    button::Style {
        background,
        text_color: if selected { ACCENT_HI } else { TEXT_DIM },
        border: border::rounded(10),
        ..Default::default()
    }
}

/// Drag-and-drop target styling is not needed, but a card that reacts to the
/// pointer does: [`card`] plus a hover tint.
pub fn card_interactive(_theme: &Theme, status: button::Status) -> button::Style {
    let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);

    button::Style {
        background: Some(if hovered { SURFACE_HI } else { SURFACE }.into()),
        text_color: TEXT,
        border: Border {
            color: if hovered { accent_tint(0.5) } else { BORDER },
            width: 1.0,
            radius: 14.0.into(),
        },
        shadow: Shadow {
            color: Color::from_rgba8(0, 0, 0, 0.35),
            offset: Vector::new(0.0, 8.0),
            blur_radius: 20.0,
        },
        ..Default::default()
    }
}

/// Progress bar for the ingest job.
pub fn progress(_theme: &Theme) -> progress_bar::Style {
    progress_bar::Style {
        background: SURFACE_HI.into(),
        bar: ACCENT.into(),
        border: border::rounded(999),
    }
}

/// The model/status badge in the top bar.
pub fn badge(_theme: &Theme) -> container::Style {
    container::Style {
        background: Some(SURFACE_HI.into()),
        border: border::rounded(999),
        ..Default::default()
    }
}

/// Darker panel used for log output.
pub fn log_panel(_theme: &Theme) -> container::Style {
    container::Style {
        background: Some(BG.into()),
        border: Border {
            color: BORDER,
            width: 1.0,
            radius: 10.0.into(),
        },
        ..Default::default()
    }
}

/// Colours used for status text.
pub fn danger_text(_theme: &Theme) -> text::Style {
    text::Style {
        color: Some(Color::from_rgb8(0xff, 0x8b, 0x8b)),
    }
}

pub fn success_text(_theme: &Theme) -> text::Style {
    text::Style {
        color: Some(Color::from_rgb8(0x7d, 0xdf, 0x9a)),
    }
}

