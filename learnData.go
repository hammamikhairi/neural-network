package neuralnetwork

type LayerLearnData struct {
	inputs         []float64
	weightedInputs []float64
	activations    []float64
	nodeValues     []float64
}

func NewLayerLearnData(layer *Layer) *LayerLearnData {
	return &LayerLearnData{
		weightedInputs: make([]float64, layer.NumNOut),
		activations:    make([]float64, layer.NumNOut),
		nodeValues:     make([]float64, layer.NumNOut),
	}
}

type NetworkLearnData struct {
	layerData []*LayerLearnData
}

func NewNetworkLearnData(layers []*Layer) *NetworkLearnData {
	layerData := make([]*LayerLearnData, len(layers))
	for i, layer := range layers {
		layerData[i] = NewLayerLearnData(layer)
	}
	return &NetworkLearnData{
		layerData: layerData,
	}
}
