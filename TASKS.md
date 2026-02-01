# RONN Development Tasks

## ✅ v0.1.0 Release Ready

All blockers resolved:
- [x] Fix `unsqueeze` bounds checking bug
- [x] Restore test suites (6,600+ lines recovered)
- [x] All 978 tests passing
- [x] Create CHANGELOG.md
- [x] Fix README badge URLs
- [x] Remove misleading status documents
- [x] Clean up outdated docs

### Nice to Have (Post-Release)
- [ ] Enable integration tests with real ONNX models
- [ ] Reduce clippy lint allows
- [ ] Add deployment documentation

---

## Project Overview

**Architecture**:
```
ronn-api       → High-level user API
ronn-core      → Core tensor operations, session management
ronn-onnx      → ONNX parsing and compatibility (44 operators)
ronn-graph     → Graph optimization pipeline (6 passes)
ronn-providers → Execution providers (CPU, GPU, BitNet, WASM, Custom)
ronn-hrm       → Hierarchical Reasoning Module (brain-inspired)
ronn-memory    → Multi-tier memory system
ronn-learning  → Continual learning engine
ronn-python    → Python bindings (PyO3)
```

---

## Future Roadmap

### v0.2.0
- Additional ONNX operators (Pad, InstanceNorm, GroupNorm, etc.)
- C/C++ FFI bindings
- Enhanced profiling (flamegraph, Chrome tracing)

### v0.3.0
- Model zoo with pre-converted popular models
- TensorBoard integration
- Distributed inference

### v1.0.0
- Production-hardened with comprehensive benchmarks
- Full ONNX opset coverage
- Enterprise features (auth, rate limiting, etc.)
