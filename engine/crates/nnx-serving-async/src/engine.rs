use crate::stream::{AsyncServingError, AsyncServingRequest, TokenOutput, TokenStream};
use nnx_serving::backend::ServingEngine;
use nnx_serving::scheduler::SchedulerStats;
use nnx_serving::sequence::{FinishReason, SequenceId};
use nnx_transformer::sampler;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use tokio::sync::{mpsc as tokio_mpsc, oneshot};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, warn};

struct SequenceRuntime {
    sender: tokio_mpsc::UnboundedSender<TokenOutput>,
    sampler: nnx_transformer::SamplerConfig,
    eos_token_id: Option<u32>,
    history: Vec<u32>,
    generated_tokens: usize,
    max_new_tokens: usize,
    rng_state: u64,
}

enum WorkerCommand {
    Submit {
        request: AsyncServingRequest,
        sender: tokio_mpsc::UnboundedSender<TokenOutput>,
        response: oneshot::Sender<Result<SequenceId, AsyncServingError>>,
    },
    Cancel {
        seq_id: SequenceId,
        response: Option<oneshot::Sender<()>>,
    },
    Stats {
        response: oneshot::Sender<Result<SchedulerStats, AsyncServingError>>,
    },
    Shutdown,
}

fn run_worker(mut engine: ServingEngine, cmd_rx: mpsc::Receiver<WorkerCommand>) {
    let mut runtimes: HashMap<SequenceId, SequenceRuntime> = HashMap::new();

    loop {
        while let Ok(command) = cmd_rx.try_recv() {
            if handle_command(&mut engine, &mut runtimes, command) {
                return;
            }
        }

        if engine.has_work() {
            match engine.step() {
                Ok(iteration) => {
                    for output in iteration.outputs {
                        let Some(runtime) = runtimes.get_mut(&output.seq_id) else {
                            continue;
                        };

                        let token_id = sampler::sample(
                            &output.logits,
                            &runtime.sampler,
                            &runtime.history,
                            &mut runtime.rng_state,
                        );
                        let is_eos = runtime.eos_token_id == Some(token_id);
                        runtime.history.push(token_id);
                        runtime.generated_tokens += 1;

                        let finish_reason = if is_eos {
                            Some(FinishReason::EndOfSequence)
                        } else if runtime.generated_tokens >= runtime.max_new_tokens {
                            Some(FinishReason::MaxTokens)
                        } else {
                            None
                        };

                        let _ = runtime.sender.send(TokenOutput {
                            seq_id: output.seq_id,
                            token_id,
                            finish_reason: finish_reason.clone(),
                        });
                        engine.on_token_generated(output.seq_id, token_id, is_eos);
                    }

                    for (seq_id, reason) in iteration.finished {
                        debug!(?seq_id, ?reason, "async serving sequence finished");
                        runtimes.remove(&seq_id);
                    }
                }
                Err(error) => {
                    warn!(%error, "async serving worker stopping after step failure");
                    return;
                }
            }

            continue;
        }

        match cmd_rx.recv() {
            Ok(command) => {
                if handle_command(&mut engine, &mut runtimes, command) {
                    return;
                }
            }
            Err(_) => return,
        }
    }
}

fn handle_command(
    engine: &mut ServingEngine,
    runtimes: &mut HashMap<SequenceId, SequenceRuntime>,
    command: WorkerCommand,
) -> bool {
    match command {
        WorkerCommand::Submit {
            request,
            sender,
            response,
        } => {
            let result = engine
                .try_add_request(request.prompt_tokens.clone(), request.max_new_tokens)
                .map_err(AsyncServingError::from)
                .map(|seq_id| {
                    runtimes.insert(
                        seq_id,
                        SequenceRuntime {
                            sender,
                            sampler: request.sampler,
                            eos_token_id: request.eos_token_id,
                            history: request.prompt_tokens,
                            generated_tokens: 0,
                            max_new_tokens: request.max_new_tokens,
                            rng_state: request.seed.max(1),
                        },
                    );
                    seq_id
                });
            let _ = response.send(result);
            false
        }
        WorkerCommand::Cancel { seq_id, response } => {
            let _ = engine.cancel_request(seq_id);
            runtimes.remove(&seq_id);
            if let Some(response) = response {
                let _ = response.send(());
            }
            false
        }
        WorkerCommand::Stats { response } => {
            let _ = response.send(Ok(engine.stats()));
            false
        }
        WorkerCommand::Shutdown => true,
    }
}

struct HandleState {
    seq_id: SequenceId,
    command_tx: mpsc::Sender<WorkerCommand>,
}

impl Drop for HandleState {
    fn drop(&mut self) {
        let _ = self.command_tx.send(WorkerCommand::Cancel {
            seq_id: self.seq_id,
            response: None,
        });
    }
}

/// Request-scoped handle returned by [`AsyncServingEngine::submit`].
#[derive(Clone)]
pub struct RequestHandle {
    state: Arc<HandleState>,
}

impl RequestHandle {
    /// Sequence id associated with this request.
    pub fn sequence_id(&self) -> SequenceId {
        self.state.seq_id
    }

    /// Cancel the request explicitly.
    pub async fn cancel(&self) -> Result<(), AsyncServingError> {
        let (tx, rx) = oneshot::channel();
        self.state
            .command_tx
            .send(WorkerCommand::Cancel {
                seq_id: self.state.seq_id,
                response: Some(tx),
            })
            .map_err(|_| AsyncServingError::WorkerStopped)?;
        rx.await.map_err(|_| AsyncServingError::WorkerStopped)?;
        Ok(())
    }
}

/// Async wrapper over the synchronous `nnx-serving` engine.
pub struct AsyncServingEngine {
    command_tx: mpsc::Sender<WorkerCommand>,
    streams: Arc<Mutex<HashMap<SequenceId, tokio_mpsc::UnboundedReceiver<TokenOutput>>>>,
    worker: Option<thread::JoinHandle<()>>,
}

impl AsyncServingEngine {
    /// Start a background worker that owns the synchronous serving engine.
    pub fn new(engine: ServingEngine) -> Self {
        let (command_tx, command_rx) = mpsc::channel();
        let streams = Arc::new(Mutex::new(HashMap::new()));
        let worker = thread::Builder::new()
            .name("nnx-serving-async".into())
            .spawn(move || run_worker(engine, command_rx))
            .expect("failed to spawn async serving worker");

        Self {
            command_tx,
            streams,
            worker: Some(worker),
        }
    }

    /// Submit a request and return immediately once the worker has admitted it.
    pub async fn submit(
        &self,
        request: AsyncServingRequest,
    ) -> Result<RequestHandle, AsyncServingError> {
        let (stream_tx, stream_rx) = tokio_mpsc::unbounded_channel();
        let (result_tx, result_rx) = oneshot::channel();

        self.command_tx
            .send(WorkerCommand::Submit {
                request,
                sender: stream_tx,
                response: result_tx,
            })
            .map_err(|_| AsyncServingError::WorkerStopped)?;

        let seq_id = result_rx
            .await
            .map_err(|_| AsyncServingError::WorkerStopped)??;
        self.streams
            .lock()
            .expect("stream registry poisoned")
            .insert(seq_id, stream_rx);

        Ok(RequestHandle {
            state: Arc::new(HandleState {
                seq_id,
                command_tx: self.command_tx.clone(),
            }),
        })
    }

    /// Take the token stream for a previously submitted sequence.
    pub fn stream(&self, seq_id: SequenceId) -> Result<TokenStream, AsyncServingError> {
        let receiver = self
            .streams
            .lock()
            .expect("stream registry poisoned")
            .remove(&seq_id)
            .ok_or(AsyncServingError::StreamNotFound(seq_id))?;
        Ok(UnboundedReceiverStream::new(receiver))
    }

    /// Fetch current scheduler statistics from the worker.
    pub async fn stats(&self) -> Result<SchedulerStats, AsyncServingError> {
        let (tx, rx) = oneshot::channel();
        self.command_tx
            .send(WorkerCommand::Stats { response: tx })
            .map_err(|_| AsyncServingError::WorkerStopped)?;
        rx.await.map_err(|_| AsyncServingError::WorkerStopped)?
    }
}

impl Drop for AsyncServingEngine {
    fn drop(&mut self) {
        let _ = self.command_tx.send(WorkerCommand::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
