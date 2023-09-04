const path = require('path');
const webpack = require('webpack');

const CopyWebpackPlugin = require('copy-webpack-plugin');
const { CleanWebpackPlugin } = require('clean-webpack-plugin');
const HtmlWebpackPlugin = require('html-webpack-plugin');


module.exports = {
    entry: [ "./src/main.ts"],
    output: {
        library: "WebSR",
        libraryTarget: "umd",
        libraryExport: "default",
        path: path.resolve(__dirname, './demo'),
        chunkFilename: 'websr.[name].js',
        filename: "websr.js"
    },
    module: {

        rules: [
            {
                test: /\.ts?$/,
                use: {
                    loader: "ts-loader",
                    options: {
                        compilerOptions:{
                            "types": ["@webgpu/types"]
                        }
                    }
                },
                exclude: /node_modules/,

            },
        ],

    },

    plugins: [

        new HtmlWebpackPlugin({
            template: 'src/demo/index.html'
        }),

        new CleanWebpackPlugin({
            cleanStaleWebpackAssets: false
        }),

        new CopyWebpackPlugin( {
            patterns: [
                { from: "src/demo/test.png", to: path.basename('test.png') },
                { from: "src/demo/anime4k-cnn-2x-s.json", to: path.basename('anime4k-cnn-2x-s.json') }
            ]
        })

    ],
    resolve: {
        extensions: [".ts", ".tsx", ".js"]
    },

    devServer: {
        static: {
            directory: path.join(__dirname, 'demo'),
        },
        compress: true,
        port: 8000,
    },

    mode: 'development'

};
