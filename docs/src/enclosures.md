# Enclosure Methods

Rigorous enclosures compute interval boxes guaranteed to contain all images of points in an input box.

## Zeroth-Order Enclosure

Evaluate `F` directly on interval arguments:

```julia
image = zeroth_order_enclosure(F, box)
```

Simple but can be loose for nonlinear functions.

## First-Order Enclosure

Uses automatic differentiation via the mean value theorem:

```julia
image = first_order_enclosure(F, box)
```

Computes `F(mid(x)) + DF(x) · (x - mid(x))` where `DF(x)` is the interval Jacobian. Significantly tighter for nonlinear maps.

## API Reference

```@docs
zeroth_order_enclosure
first_order_enclosure
```
