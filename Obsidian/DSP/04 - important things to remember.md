
1. initial set up of pre-trained AI models are simplified:
	1. meaning in the later stages of development they need to have all blocks implemented for best performance and accuracy 

2. currently the models are using dummy weights for the model testing, later on post development I will download train model weights, then load them.
	1. update model implementations to their full versions and re-enable tests



For a production system, I need to eventually:

1. Replace the simplified test implementations with full implementations once I have proper weights
2. Re-enable the skipped tests
3. Add mock weights specifically for testing