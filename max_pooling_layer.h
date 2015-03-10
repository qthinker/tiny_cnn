/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY 
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND 
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

	Add by WangZiJie 2014/11
*/
#pragma once
#include "util.h"
#include "partial_connected_layer.h"

namespace tiny_cnn {


template<typename N, typename Activation>
class max_pooling_layer : public partial_connected_layer<N, Activation> {
public:
    typedef partial_connected_layer<N, Activation> Base;
    typedef typename Base::Optimizer Optimizer;

    max_pooling_layer(int in_width, int in_height, int in_channels, int pooling_size)
    : partial_connected_layer<N, Activation>(
     in_width * in_height * in_channels, 
     in_width * in_height * in_channels / (pooling_size * pooling_size), 
     in_channels, in_channels/*, 1.0 / (pooling_size * pooling_size)*/),
     in_(in_width, in_height, in_channels), 
     out_(in_width/pooling_size, in_height/pooling_size, in_channels)
    {
        if ((in_width % pooling_size) || (in_height % pooling_size)) 
            throw nn_error("width/height must be multiples of pooling size");
        init_connection(pooling_size);
		out2maxi_.resize(out_size_, 0);
    }

	virtual const vec_t& forward_propagation(const vec_t& in, int index) {

        for_(this->parallelize_, 0, this->out_size_, [&](const blocked_range& r) {
            for (int i = r.begin(); i < r.end(); i++) {
                const wi_connections& connections = out2wi_[i];
				/*
                float_t a = 0.0;
			
                for (auto connection : connections)// 13.1%
                    a += this->W_[connection.first] * in[connection.second]; // 3.2%

                a *= scale_factor_;
                a += this->b_[out2bias_[i]];
				*/
				float_t a = -10000.0;
				for (auto connection : connections)
				{
					if(a < in[connection.second])
					{
						a = this->W_[connection.first] * in[connection.second];	//每一个输出单元等于链接的输入的最大值
						out2maxi_[i] = connection.second;	//保存最大的输入的索引
					}
				}
				a *= scale_factor_;
                a += this->b_[out2bias_[i]];
                this->output_[index][i] = this->a_.f(a); // 9.6%
            }
        });
        return this->next_ ? this->next_->forward_propagation(this->output_[index], index) : this->output_[index]; // 15.6%
    }

	virtual const vec_t& back_propagation(const vec_t& current_delta, int index) {
        const vec_t& prev_out = this->prev_->output(index);
        const activation::function& prev_h = this->prev_->activation_function();
        vec_t& prev_delta = this->prev_delta_[index];

        for_(this->parallelize_, 0, this->in_size_, [&](const blocked_range& r) {
            for (int i = r.begin(); i != r.end(); i++) {
                const wo_connections& connections = in2wo_[i];
                float_t delta = 0.0;

                for (auto connection : connections) 
				{
					if(out2maxi_[connection.second] == i)  //max-pooling 只在max的位置计算delta
						delta += this->W_[connection.first] * current_delta[connection.second]; 
				}

                prev_delta[i] = delta * scale_factor_ * prev_h.df(prev_out[i]); // 2.1%
            }
        });

        for_(this->parallelize_, 0, weight2io_.size(), [&](const blocked_range& r) {
            for (int i = r.begin(); i < r.end(); i++) {
                const io_connections& connections = weight2io_[i];
                float_t diff = 0.0;

                for (auto connection : connections) // 11.9%
				{
					if(connection.first == out2maxi_[connection.second]) //max-pooling 只在max位置计算
						diff += prev_out[connection.first] * current_delta[connection.second];
				}

                this->dW_[index][i] += diff * scale_factor_;
            }
        });

        for (size_t i = 0; i < bias2out_.size(); i++) {
            const std::vector<int>& outs = bias2out_[i];
            float_t diff = 0.0;

            for (auto o : outs)
                diff += current_delta[o];    

            this->db_[index][i] += diff;
        } 

        return this->prev_->back_propagation(this->prev_delta_[index], index);
    }

	virtual const vec_t& back_propagation_2nd(const vec_t& current_delta2) {
        const vec_t& prev_out = this->prev_->output(0);
        const activation::function& prev_h = this->prev_->activation_function();

        for (size_t i = 0; i < weight2io_.size(); i++) {
            const io_connections& connections = weight2io_[i];
            float_t diff = 0.0;

            for (auto connection : connections)
			{
				if(connection.first == out2maxi_[connection.second]) //max 位置
					diff += prev_out[connection.first] * prev_out[connection.first] * current_delta2[connection.second];
			}

            diff *= scale_factor_ * scale_factor_;
            this->Whessian_[i] += diff;
        }

        for (size_t i = 0; i < bias2out_.size(); i++) {
            const std::vector<int>& outs = bias2out_[i];
            float_t diff = 0.0;

            for (auto o : outs)
                diff += current_delta2[o];    

            this->bhessian_[i] += diff;
        }

        for (int i = 0; i < this->in_size_; i++) {
            const wo_connections& connections = in2wo_[i];
            this->prev_delta2_[i] = 0.0;

            for (auto connection : connections) 
			{
				if(i == out2maxi_[connection.second]) //max位置
					this->prev_delta2_[i] += this->W_[connection.first] * this->W_[connection.first] * current_delta2[connection.second];
			}
            this->prev_delta2_[i] *= scale_factor_ * scale_factor_ * prev_h.df(prev_out[i]) * prev_h.df(prev_out[i]);
        }
        return this->prev_->back_propagation_2nd(this->prev_delta2_);
    }

private:
    void init_connection(int pooling_size) {
        for (int c = 0; c < in_.depth_; c++) 
            for (int y = 0; y < in_.height_; y += pooling_size)
                for (int x = 0; x < in_.width_; x += pooling_size)
                    connect_kernel(pooling_size, x, y, c);


        for (int c = 0; c < in_.depth_; c++) 
            for (int y = 0; y < out_.height_; y++)
                for (int x = 0; x < out_.width_; x++)
                    this->connect_bias(c, out_.get_index(x, y, c));
    }

    void connect_kernel(int pooling_size, int x, int y, int inc) {
        for (int dy = 0; dy < pooling_size; dy++)
            for (int dx = 0; dx < pooling_size; dx++)
                this->connect_weight(
                    in_.get_index(x + dx, y + dy, inc), 
                    out_.get_index(x / pooling_size, y / pooling_size, inc),
                    inc);
    }

    tensor3d in_;
    tensor3d out_;
	std::vector<int> out2maxi_;
};

} // namespace tiny_cnn
