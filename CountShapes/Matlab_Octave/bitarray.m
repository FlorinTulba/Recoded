%{
    Part of the CountShapes project.

    @2017 Florin Tulba (florintulba@yahoo.com)
%}

classdef bitarray < handle
    %BITARRAY provides the functionality of 'dynamic_bitset' from Boost (C++)
    %   Using an array of 'uint32' to be able to store enough bits.
    %   Applying 'bitset', 'bitand', ... to each of the 'uint32' values
    
    properties
        N       % actual bits count
        data    % the array of required 'uint32' values. Its length is ceil(N, 32)
        
        % Most significant bits are in data(end), which contains only        
        remN    % = floor(N, 32) bits
        
        countedOnes    % the count of bits on value 1
    end
    
    methods
        % Create a bitarray of initial size 'initialSz' with all bits on 0
        function ba = bitarray(initialSz)
            if nargin > 0
                ba.N = uint16(initialSz);
                ba.data = zeros(ceil(initialSz / 32), 1, 'uint32');
                ba.remN = uint16(1 + mod(initialSz-1, 32));
            else
                ba.N = uint16(0);
                ba.remN = uint16(0);
                ba.data = zeros(0, 1, 'uint32');
            end
            ba.countedOnes = uint16(0);
        end
        
        % Prepend one bit to the bitarray as the new most significant bit
        % This will actually be appended in data(end)!
        function prepend(self, bitValue)
            if nargin < 2 || (bitValue ~= 0 && bitValue ~= 1)
                bitValue = 0;
            end
            bitValue = uint32(bitValue);
            if mod(self.remN, 32) == 0 % data array must be extended with one item
                self.data = [self.data ; bitValue];
                self.remN = uint16(1);
            else
                self.remN = self.remN + uint16(1);
                self.data(end) = bitset(self.data(end), self.remN, bitValue);
            end
            self.N = self.N + uint16(1);
            if bitValue == 1
                self.countedOnes = self.countedOnes + uint16(1);
            end
        end
        
        % Complementing current data
        function compl(self)
            self.data = bitcmp(self.data);
            self.countedOnes = self.N - self.countedOnes;
        end
        
        % Performs 'and' on bits with a different <bitarray>
        function newBa = and(self, otherBa)
            newBa = bitarray(self.N);
            if nargin > 1 && self.N > 0 && self.N == otherBa.N
                newBa.data = bitand(self.data, otherBa.data);
                dataLen = length(self.data);
                for i = 1 : (dataLen - 1)
                    newBa.countedOnes = newBa.countedOnes + ...
                        nnz(bitget(newBa.data(i), (1:32)'));
                end
                newBa.countedOnes = newBa.countedOnes + ...
                    nnz(bitget(newBa.data(dataLen), (1:self.remN)'));
            end
        end
        
        % Performs 'or' on bits with a different <bitarray>
        function newBa = or(self, otherBa)
            newBa = bitarray(self.N);
            if nargin > 1 && self.N > 0 && self.N == otherBa.N
                newBa.data = bitor(self.data, otherBa.data);
                dataLen = length(self.data);
                for i = 1 : (dataLen - 1)
                    newBa.countedOnes = newBa.countedOnes + ...
                        nnz(bitget(newBa.data(i), (1:32)'));
                end
                newBa.countedOnes = newBa.countedOnes + ...
                    nnz(bitget(newBa.data(dataLen), (1:self.remN)'));
            end
        end
        
        % Setting the bit bitIdx (1..N) to bitValue, which defaults to 1
        function set(self, bitIdx, bitValue)
            if nargin > 1 && bitIdx >= 1 && bitIdx <= self.N
                if nargin < 3 || (bitValue ~= 0 && bitValue ~= 1)
                    bitValue = 1;
                end
                bitValue = uint32(bitValue);
                dataIdx = ceil(double(bitIdx) / 32);
                actualBitIdx = 1 + mod(bitIdx-1, 32);
                previousBit = bitget(self.data(dataIdx), actualBitIdx);
                if previousBit ~= bitValue
                    self.data(dataIdx) = ...
                        bitset(self.data(dataIdx), actualBitIdx, bitValue);
                    if bitValue == 1
                        self.countedOnes = self.countedOnes + uint16(1);
                    else
                        self.countedOnes = self.countedOnes - uint16(1);
                    end
                end
            end
        end
        
        % Getting the bit bitIdx (1..N)
        function b = get(self, bitIdx)
            if nargin > 1 && bitIdx >= 1 && bitIdx <= self.N
                dataIdx = ceil(double(bitIdx) / 32);
                actualBitIdx = 1 + mod(bitIdx-1, 32);
                b = uint8(bitget(self.data(dataIdx), actualBitIdx));
            else
                b = uint8(0);
            end
        end
        
        % Get the positions of the bits set on 1
        function ind1 = indicesOf1(self)
            ind1 = zeros(self.countedOnes, 1, 'uint16');
            nextPos = 1;
            dataLen = length(self.data);
            for i = 1 : (dataLen - 1)
                nextIndices = 32*(i-1) + ...
                    find(bitget(self.data(i), (1:32)'));
                lastPos = nextPos + length(nextIndices) - 1;
                ind1(nextPos:lastPos) = uint16(nextIndices);
                nextPos = lastPos + 1;
            end
            ind1(nextPos:end) = uint16(32*(dataLen-1) + ...
                            find(bitget(self.data(dataLen), (1:self.remN)')));
        end
        
        % 'to string' method
        function Str = str(self, reversed)
            Str = sprintf('');
            if self.N > 0
                for i = 1 : (numel(self.data)-1)
                    Str = sprintf('%032s%s', dec2bin(self.data(i)), Str);
                end
                MostSignificantPart = ...
                    dec2bin(bitand(self.data(end), ...
                                    bitsrl(intmax('uint32'), 32-self.remN)));
                Padding = repmat('0', 1, ...
                                self.remN - length(MostSignificantPart));
                Str = sprintf('%s%s%s', Padding, MostSignificantPart, Str);
                
                if nargin > 1 && reversed
                    Str = Str(end:-1:1);
                end
            end
        end
    end
end

