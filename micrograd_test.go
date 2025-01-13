package main

import (
	"fmt"
	"testing"
)

func TestBasicAdd(t *testing.T) {
	a := &Value{Data: 1.0, Grad: 0.0}
	b := &Value{Data: 2.0, Grad: 0.0}
	c := Add(a, b)

	fmt.Println(c.Data)
	if c.Data != 3.0 {
		t.Errorf("Expected 3.0, got %f", c.Data)
	}
}

func TestBasicMul(t *testing.T) {
	a := &Value{Data: 2.0, Grad: 0.0}
	b := &Value{Data: 3.0, Grad: 0.0}
	c := Mul(a, b)

	fmt.Println(c.Data)
	if c.Data != 6.0 {
		t.Errorf("Expected 6.0, got %f", c.Data)
	}
}

func TestBasicMulAdd(t *testing.T) {
	a := &Value{Data: 2.0, Grad: 0.0, Label: "a"}
	b := &Value{Data: 3.0, Grad: 0.0, Label: "b"}
	c := Mul(a, b)
	c.Label = "c"
	d := Add(c, a)
	d.Label = "d"
	d.Print()
	if d.Data != 8.0 {
		t.Errorf("Expected 8.0, got %f", d.Data)
	}
}

func TestBasicGrad(t *testing.T) {

	a := &Value{Data: 2.0, Grad: 0.0, Label: "a"}
	b := &Value{Data: -3.0, Grad: 0.0, Label: "b"}
	c := &Value{Data: 10.0, Grad: 0.0, Label: "c"}

	e := Mul(a, b)
	e.Label = "e"

	d := Add(e, c)
	d.Label = "d"

	f := &Value{Data: -2.0, Grad: 0.0, Label: "f"}
	L := Mul(d, f)
	L.Label = "L"
	L.Print()

	if L.Data != -8 {
		t.Errorf("Expected -8.0, got %f", L.Data)
	}
	if L.Grad != -20.0 {
		t.Errorf("Expected -20.0, got %f", L.Grad)
	}

}
