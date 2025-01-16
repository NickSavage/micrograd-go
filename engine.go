package main

import (
	"fmt"
	"math/rand/v2"
)

type Neuron struct {
	W []*Value
	B *Value
}

type Layer struct {
	Neurons []*Neuron
}

type MLP struct {
	Layers []*Layer
}

type MLPState struct {
	NumInputs   int   `json:"num_inputs"`
	LayerSizes  []int `json:"layer_sizes"`
	LayerStates [][]struct {
		Weights []float64 `json:"weights"`
		Bias    float64   `json:"bias"`
	} `json:"layer_states"`
}

func NewNeuron(numInputs int) *Neuron {
	w := make([]*Value, numInputs)
	for i := range w {
		w[i] = &Value{
			Data:     rand.Float64()*2 - 1,
			Grad:     0.0,
			Children: nil,
			Op:       "",
			Label:    fmt.Sprintf("w%d", i),
		}
	}
	b := &Value{
		Data:     rand.Float64()*2 - 1,
		Grad:     0.0,
		Children: nil,
		Op:       "",
		Label:    "b",
	}
	return &Neuron{W: w, B: b}
}

func (n *Neuron) Print() {
	fmt.Printf("Neuron(W: %v, B: %v)\n", n.W, n.B)
}

func (n *Neuron) Call(x []*Value) *Value {
	sum := n.B
	for i, w := range n.W {
		sum = Add(sum, Mul(w, x[i]))
	}
	out := Tanh(sum)
	return out
}

func (n *Neuron) Parameters() []*Value {
	return append(n.W, n.B)
}

func NewLayer(numInputs, numOutputs int) *Layer {
	neurons := make([]*Neuron, numOutputs)
	for i := range neurons {
		neurons[i] = NewNeuron(numInputs)
	}
	return &Layer{Neurons: neurons}
}

func (l *Layer) Print() {
	for _, neuron := range l.Neurons {
		neuron.Print()
	}
}

func (l *Layer) Call(x []*Value) []*Value {
	outputs := make([]*Value, len(l.Neurons))
	for i, neuron := range l.Neurons {
		outputs[i] = neuron.Call(x)
	}
	return outputs
}

func (l *Layer) Parameters() []*Value {
	params := []*Value{}
	for _, neuron := range l.Neurons {
		params = append(params, neuron.Parameters()...)
	}
	return params
}

func NewMLP(numInputs int, layerSizes []int) *MLP {
	layers := make([]*Layer, len(layerSizes))

	// current size represents the number of inputs for the next layer
	currentSize := numInputs

	for i, outputSize := range layerSizes {
		layers[i] = NewLayer(currentSize, outputSize)
		currentSize = outputSize
	}

	return &MLP{Layers: layers}
}

func (m *MLP) Print() {
	for _, layer := range m.Layers {
		layer.Print()
	}
}

func (m *MLP) Call(x []*Value) []*Value {
	for _, layer := range m.Layers {
		x = layer.Call(x)
	}
	return x
}

func (m *MLP) Parameters() []*Value {
	params := []*Value{}
	for _, layer := range m.Layers {
		params = append(params, layer.Parameters()...)
	}
	return params
}
