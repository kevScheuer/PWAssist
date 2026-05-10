
## Phase 1
### Structure
My source code this project will be based on is bloated and hard to track. I need to start with a few simple raw csv files produced from my actual analysis, and think of a sensible way to structure my project so I can transform them into plots of mass-independent pwa. **Its okay to start small**.

*When making updates, use uv version ability. Try to see how to align it with github repo version*

### Setup
Create a few modules that loosely describe what they will handle

#### Preprocessing
I like the idea of having a preprocessor with clearly chunked steps, that then runs each step in a sequence and prints out the time taken for each.

## Phase 2 
### Basic Implementation
I should be able to have my individual csvs be collected, preprocessed, loaded, and a basic plot made

### First tests
Understand how to build tests for my components

### Iterate On Other Components
Then add the other plotters, some of their methods, make a test, and so on. Make sure all of these plotters follow the idea of being able to receive an already created ax or set of axes

## Phase 3 
### Documentation
Build sphinx documentation

### Link documentation to github repo
