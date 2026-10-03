//! Audio playback backend for the transcript player.
//!
//! `rodio`'s output stream is `!Send`, so the sink/player live on a dedicated
//! thread. The UI talks to it over an mpsc channel and reads the playhead from
//! atomics (no locking, no UI stalls).

use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::time::Duration;

use rodio::{Decoder, DeviceSinkBuilder, Player};

#[derive(Debug)]
pub enum Cmd {
    Toggle,
    Seek(f32),
    Volume(f32),
    Quit,
}

pub struct AudioHandle {
    tx: Sender<Cmd>,
    pos_ms: Arc<AtomicU64>,
    playing: Arc<AtomicBool>,
    failed: Arc<AtomicBool>,
}

impl AudioHandle {
    pub fn spawn(path: PathBuf) -> Self {
        let (tx, rx) = mpsc::channel::<Cmd>();
        let pos_ms = Arc::new(AtomicU64::new(0));
        let playing = Arc::new(AtomicBool::new(false));
        let failed = Arc::new(AtomicBool::new(false));

        let thread_pos = pos_ms.clone();
        let thread_playing = playing.clone();
        let thread_failed = failed.clone();

        std::thread::Builder::new()
            .name("audio".into())
            .spawn(move || {
                let mut sink = match DeviceSinkBuilder::open_default_sink() {
                    Ok(sink) => sink,
                    Err(err) => {
                        eprintln!("audio: cannot open output device: {err}");
                        thread_failed.store(true, Ordering::Relaxed);
                        return;
                    }
                };
                sink.log_on_drop(false); // no scary stderr message on exit
                let player = Player::connect_new(sink.mixer());
                let mut loaded = false;

                // keep `sink` alive for the whole thread
                let _keep_alive = &sink;

                loop {
                    match rx.recv_timeout(Duration::from_millis(25)) {
                        Ok(Cmd::Toggle) => {
                            if !loaded {
                                loaded = append(&player, &path, &thread_failed);
                            } else if player.is_paused() {
                                player.play();
                            } else {
                                player.pause();
                            }
                        }
                        Ok(Cmd::Seek(secs)) => {
                            if !loaded {
                                loaded = append(&player, &path, &thread_failed);
                            }
                            if let Err(err) = player.try_seek(Duration::from_secs_f32(secs.max(0.0)))
                            {
                                eprintln!("audio: seek failed: {err}");
                            }
                            player.play();
                        }
                        Ok(Cmd::Volume(volume)) => player.set_volume(volume.clamp(0.0, 1.0)),
                        Ok(Cmd::Quit) | Err(RecvTimeoutError::Disconnected) => break,
                        Err(RecvTimeoutError::Timeout) => {}
                    }

                    let pos = player.get_pos();
                    thread_pos.store(pos.as_millis() as u64, Ordering::Relaxed);
                    thread_playing.store(
                        !player.is_paused() && !player.empty(),
                        Ordering::Relaxed,
                    );

                    // Track finished: rewind so "play" starts over from the top.
                    if loaded && player.empty() {
                        let _ = player.try_seek(Duration::ZERO);
                        loaded = false;
                    }
                }
            })
            .expect("spawn audio thread");

        Self { tx, pos_ms, playing, failed }
    }

    /// Playhead in seconds.
    pub fn position(&self) -> f32 {
        self.pos_ms.load(Ordering::Relaxed) as f32 / 1000.0
    }

    pub fn is_playing(&self) -> bool {
        self.playing.load(Ordering::Relaxed)
    }

    pub fn failed(&self) -> bool {
        self.failed.load(Ordering::Relaxed)
    }

    pub fn send(&self, cmd: Cmd) {
        let _ = self.tx.send(cmd);
    }
}

impl Drop for AudioHandle {
    fn drop(&mut self) {
        let _ = self.tx.send(Cmd::Quit);
    }
}

fn append(player: &Player, path: &PathBuf, failed: &AtomicBool) -> bool {
    match File::open(path).map_err(|e| e.to_string()).and_then(|file| {
        Decoder::try_from(file).map_err(|e| e.to_string())
    }) {
        Ok(source) => {
            player.append(source);
            player.play();
            true
        }
        Err(err) => {
            eprintln!("audio: cannot decode {}: {err}", path.display());
            failed.store(true, Ordering::Relaxed);
            false
        }
    }
}
