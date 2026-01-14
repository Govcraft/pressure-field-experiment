//! Tick driver actor for receiving TickComplete from the kernel coordinator.
//!
//! This simple actor bridges the kernel's actor-based coordination with the
//! experiment harness by forwarding TickComplete results to an mpsc channel.

use acton_reactive::prelude::*;
use tokio::sync::{mpsc, oneshot};

use survival_kernel::kernel::TickResult;
use survival_kernel::messages::{PatchActorsReady, TickComplete};

/// State for the tick driver actor.
#[derive(Default, Clone)]
pub struct TickDriverState {
    /// Channel sender for forwarding tick results
    pub tx: Option<mpsc::Sender<TickResult>>,
}

impl std::fmt::Debug for TickDriverState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TickDriverState")
            .field("has_tx", &self.tx.is_some())
            .finish()
    }
}

/// Actor that receives TickComplete messages and forwards them to a channel.
///
/// The experiment harness creates this actor and registers it with the
/// coordinator via RegisterTickDriver. When the coordinator completes a tick,
/// it sends TickComplete to this actor, which forwards the result to the
/// mpsc channel for the experiment harness to receive.
pub struct TickDriverActor {
    /// Channel sender for forwarding tick results
    tx: mpsc::Sender<TickResult>,
}

impl TickDriverActor {
    /// Create a new tick driver actor with the given channel sender.
    pub fn new(tx: mpsc::Sender<TickResult>) -> Self {
        Self { tx }
    }

    /// Spawn the actor in the runtime.
    ///
    /// Returns the actor handle which should be sent to the coordinator
    /// via RegisterTickDriver.
    pub async fn spawn(self, runtime: &mut ActorRuntime) -> ActorHandle {
        let mut actor = runtime.new_actor_with_name::<TickDriverState>("TickDriver".to_string());

        actor.model.tx = Some(self.tx);

        actor.act_on::<TickComplete>(|actor, context| {
            let result = context.message().result.clone();
            let tx = actor.model.tx.clone();

            Reply::pending(async move {
                if let Some(tx) = tx {
                    // Ignore send errors - receiver may have been dropped
                    let _ = tx.send(result).await;
                }
            })
        });

        actor.start().await
    }
}

/// State for the startup waiter actor.
#[derive(Default, Clone)]
pub struct StartupWaiterState {
    /// Oneshot sender for signaling when actors are ready (wrapped for Clone)
    pub tx: std::sync::Arc<std::sync::Mutex<Option<oneshot::Sender<usize>>>>,
}

impl std::fmt::Debug for StartupWaiterState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StartupWaiterState")
            .field("has_tx", &"<mutex>")
            .finish()
    }
}

/// Actor that waits for PatchActorsReady broadcast and signals completion.
///
/// This actor subscribes to PatchActorsReady via broker and forwards the
/// registration count to a oneshot channel, allowing the experiment to wait
/// until all actors are registered before starting ticks.
pub struct StartupWaiter {
    /// Oneshot sender for signaling completion
    tx: oneshot::Sender<usize>,
}

impl StartupWaiter {
    /// Create a new startup waiter with the given oneshot sender.
    pub fn new(tx: oneshot::Sender<usize>) -> Self {
        Self { tx }
    }

    /// Spawn the actor in the runtime.
    ///
    /// The actor subscribes to PatchActorsReady broadcasts and forwards
    /// the registered count to the oneshot channel.
    pub async fn spawn(self, runtime: &mut ActorRuntime) {
        let mut actor = runtime.new_actor_with_name::<StartupWaiterState>("StartupWaiter".to_string());

        // Store the oneshot sender wrapped in Arc<Mutex>
        actor.model.tx = std::sync::Arc::new(std::sync::Mutex::new(Some(self.tx)));

        // Subscribe to PatchActorsReady broadcasts before starting
        actor.handle().subscribe::<PatchActorsReady>().await;

        actor.act_on::<PatchActorsReady>(|actor, context| {
            let count = context.message().registered_count;
            // Take the sender out (can only send once)
            if let Ok(mut guard) = actor.model.tx.lock() {
                if let Some(tx) = guard.take() {
                    let _ = tx.send(count);
                }
            }
            Reply::ready()
        });

        actor.start().await;
    }
}
