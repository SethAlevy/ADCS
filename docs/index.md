# Code Documentation

Below you can find the generated API documentation grouped by functional areas of the project.

## Setup

Handles external inputs to the simulation:
- TLE (Two-Line Element Set) describing the orbit.
- initial_settings.json containing simulation, satellite, sensor, actuator, and control parameters.
Both resources have dedicated reader classes.

::: setup.initial_settings
::: setup.two_line_element

## Spacecraft

Integrates satellite subsystems (sensors and actuators), exposes high‑level state, and performs control and mode management (detumbling or pointing).

::: spacecraft.satellite
::: spacecraft.actuator
::: spacecraft.sensors

## Core

Provides foundational utilities: math helpers, coordinate and frame transformations, logging, and state vector management.

::: core.state
::: core.transformations
::: core.utilities

## Visualizations

Generates plots (matplotlib for static figures, plotly for interactive and live charts). Enables quick inspection of simulation progress.

::: visualizations.visualizations

